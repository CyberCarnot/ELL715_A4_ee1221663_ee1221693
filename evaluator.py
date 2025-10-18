import os
import pickle
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cosine
from scipy.linalg import logm, sqrtm, inv
from pyeer.eer_info import get_eer_stats
from pyeer.plot import plot_eer_stats
from feature_extractor import SilhouetteTunnelFeatureExtractor

CACHE_DIR = "./cache"
FEATURES_DIR = "./cache/features"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# ==============================================================================
# AIRM DISTANCE COMPUTATION
# ==============================================================================

def compute_airm_distance(cov1, cov2):
    """
    Compute Affine-Invariant Riemannian Metric (AIRM) distance between two covariance matrices.
    
    AIRM is the geodesic distance on the manifold of Symmetric Positive Definite (SPD) matrices.
    Formula: d_AIRM(Σ1, Σ2) = ||log(Σ1^(-1/2) * Σ2 * Σ1^(-1/2))||_F
    
    Where ||.||_F is the Frobenius norm and log is the matrix logarithm.
    
    Args:
        cov1 (np.ndarray): First covariance matrix (D x D)
        cov2 (np.ndarray): Second covariance matrix (D x D)
    
    Returns:
        float: AIRM distance (always >= 0)
    """
    try:
        # Compute inverse square root of cov1: Σ1^(-1/2)
        sqrt_inv_cov1 = inv(sqrtm(cov1))
        
        # Compute the matrix product: Σ1^(-1/2) * Σ2 * Σ1^(-1/2)
        product = sqrt_inv_cov1 @ cov2 @ sqrt_inv_cov1
        
        # Compute matrix logarithm
        log_product = logm(product)
        
        # Compute Frobenius norm: ||log(product)||_F
        distance = np.linalg.norm(log_product, 'fro')
        
        # Handle numerical issues (ensure non-negative)
        distance = max(0.0, np.real(distance))
        
        return float(distance)
    
    except Exception as e:
        # If computation fails (e.g., non-positive-definite matrix), return large distance
        print(f"[WARN] AIRM computation failed: {str(e)}")
        return float('inf')


def reconstruct_covariance_from_descriptor(descriptor, feature_dim=14):
    """
    Reconstruct covariance matrix from the upper triangle descriptor.
    
    The descriptor is the upper triangle of a symmetric covariance matrix.
    We reconstruct it by filling in the lower triangle using symmetry.
    
    Args:
        descriptor (np.ndarray): Upper triangle descriptor (105-dim for 14 features)
        feature_dim (int): Number of features (14 for this application)
    
    Returns:
        np.ndarray: Reconstructed covariance matrix (feature_dim x feature_dim)
    """
    cov_matrix = np.zeros((feature_dim, feature_dim), dtype=np.float32)
    
    # Fill upper triangle from descriptor
    idx = 0
    for i in range(feature_dim):
        for j in range(i, feature_dim):
            cov_matrix[i, j] = descriptor[idx]
            idx += 1
    
    # Fill lower triangle using symmetry
    for i in range(feature_dim):
        for j in range(i):
            cov_matrix[i, j] = cov_matrix[j, i]
    
    return cov_matrix


def compute_descriptor_distance_airm(desc1, desc2, mode, feature_dim=14):
    """
    Compute distance between two descriptors using AIRM for covariance matrices.
    
    For 'baseline' and 'morph_cov' modes: Reconstruct covariance matrices and use AIRM
    For 'temporal' and 'morph_temp' modes: Average AIRM distances across temporal partitions
    
    Args:
        desc1, desc2: Descriptors (105-dim for cov, 735-dim for temporal)
        mode (str): 'baseline', 'temporal', 'morph_cov', or 'morph_temp'
        feature_dim (int): Number of features (14)
    
    Returns:
        float: Distance score
    """
    
    if mode == "baseline":
        # Both are single 105-dim covariance descriptors
        cov1 = reconstruct_covariance_from_descriptor(desc1, feature_dim)
        cov2 = reconstruct_covariance_from_descriptor(desc2, feature_dim)
        return compute_airm_distance(cov1, cov2)
    
    elif mode == "temporal":
        # Both are 735-dim temporal hierarchy descriptors
        # 735 = 105 (level 1) + 210 (level 2) + 420 (level 3)
        # Each partition is 105-dim covariance
        distances = []
        
        # Compute AIRM for each partition pair and average
        idx = 0
        for level in range(1, 4):
            num_partitions = 2**(level - 1)
            for _ in range(num_partitions):
                cov1 = reconstruct_covariance_from_descriptor(
                    desc1[idx:idx+105], feature_dim
                )
                cov2 = reconstruct_covariance_from_descriptor(
                    desc2[idx:idx+105], feature_dim
                )
                distances.append(compute_airm_distance(cov1, cov2))
                idx += 105
        
        # Return average distance across all temporal partitions
        return np.mean(distances) if distances else float('inf')
    
    elif mode == "morph_cov":
        # Both are lists of 4 covariance descriptors (full + 3 subs)
        if isinstance(desc1, list) and isinstance(desc2, list):
            distances = []
            for d1, d2 in zip(desc1, desc2):
                cov1 = reconstruct_covariance_from_descriptor(d1, feature_dim)
                cov2 = reconstruct_covariance_from_descriptor(d2, feature_dim)
                distances.append(compute_airm_distance(cov1, cov2))
            return np.mean(distances)
        else:
            # Fallback if not lists
            cov1 = reconstruct_covariance_from_descriptor(desc1, feature_dim)
            cov2 = reconstruct_covariance_from_descriptor(desc2, feature_dim)
            return compute_airm_distance(cov1, cov2)
    
    elif mode == "morph_temp":
        # Both are lists of 4 temporal hierarchy descriptors (full + 3 subs)
        if isinstance(desc1, list) and isinstance(desc2, list):
            all_distances = []
            for d1, d2 in zip(desc1, desc2):
                # Each d1, d2 is 735-dim temporal descriptor
                # Compute AIRM for each partition
                idx = 0
                for level in range(1, 4):
                    num_partitions = 2**(level - 1)
                    for _ in range(num_partitions):
                        cov1 = reconstruct_covariance_from_descriptor(
                            d1[idx:idx+105], feature_dim
                        )
                        cov2 = reconstruct_covariance_from_descriptor(
                            d2[idx:idx+105], feature_dim
                        )
                        all_distances.append(compute_airm_distance(cov1, cov2))
                        idx += 105
            return np.mean(all_distances) if all_distances else float('inf')
        else:
            # Fallback: treat as single temporal descriptors
            distances = []
            idx = 0
            for level in range(1, 4):
                num_partitions = 2**(level - 1)
                for _ in range(num_partitions):
                    cov1 = reconstruct_covariance_from_descriptor(
                        desc1[idx:idx+105], feature_dim
                    )
                    cov2 = reconstruct_covariance_from_descriptor(
                        desc2[idx:idx+105], feature_dim
                    )
                    distances.append(compute_airm_distance(cov1, cov2))
                    idx += 105
            return np.mean(distances) if distances else float('inf')
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ==============================================================================
# DATASET SCANNING (Structure only, no feature extraction)
# ==============================================================================

def scan_dataset_structure(base_path):
    """
    Scan the dataset directory structure without extracting features.
    
    Args:
        base_path (str): Root path of the dataset.
    
    Returns:
        dict: {(subj, gesture, inst): inst_path} mapping for all gesture instances.
    """
    dataset_map = {}
    
    for subj in sorted(os.listdir(base_path)):
        subj_path = os.path.join(base_path, subj)
        if not os.path.isdir(subj_path) or not subj.isdigit():
            continue
            
        for gesture in sorted(os.listdir(subj_path)):
            gesture_path = os.path.join(subj_path, gesture)
            if not os.path.isdir(gesture_path) or gesture.startswith('.') or gesture in ['__pycache__', 'cache']:
                continue
                
            for inst in sorted(os.listdir(gesture_path)):
                inst_path = os.path.join(gesture_path, inst)
                if os.path.isdir(inst_path) and not inst.startswith('.') and inst not in ['__pycache__', 'cache']:
                    dataset_map[(subj, gesture, inst)] = inst_path
    
    return dataset_map


# ==============================================================================
# PROGRESS TRACKING AND RESUME FUNCTIONALITY
# ==============================================================================

def get_progress_file():
    """Get the path to the progress tracking file."""
    return os.path.join(CACHE_DIR, "processing_progress.pkl")


def load_progress():
    """
    Load processing progress from disk.
    
    Returns:
        set: Set of completed (subj, gesture, inst) keys
    """
    progress_file = get_progress_file()
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            return pickle.load(f)
    return set()


def save_progress(completed_keys):
    """
    Save processing progress to disk.
    
    Args:
        completed_keys (set): Set of completed (subj, gesture, inst) keys
    """
    progress_file = get_progress_file()
    with open(progress_file, "wb") as f:
        pickle.dump(completed_keys, f)


def get_feature_matrix_path(subj, gesture, inst, matrix_type):
    """
    Get the file path for a specific feature matrix.
    
    Args:
        subj (str): Subject ID
        gesture (str): Gesture type
        inst (str): Instance name
        matrix_type (str): 'full', 'sub_1', 'sub_2', or 'sub_3'
    
    Returns:
        str: Path to the feature matrix file
    """
    safe_gesture = gesture.replace(" ", "_").replace("/", "_")
    filename = f"{subj}_{safe_gesture}_{inst}_{matrix_type}.npz"
    return os.path.join(FEATURES_DIR, filename)


def save_feature_matrices(subj, gesture, inst, feature_matrices):
    """
    Save all feature matrices for a gesture instance.
    
    Args:
        subj (str): Subject ID
        gesture (str): Gesture type
        inst (str): Instance name
        feature_matrices (dict): Dictionary with 'full', 'sub_1', 'sub_2', 'sub_3' keys
    """
    for matrix_type, (F, F_norm) in feature_matrices.items():
        filepath = get_feature_matrix_path(subj, gesture, inst, matrix_type)
        if F is not None and F_norm is not None:
            np.savez_compressed(filepath, F=F, F_norm=F_norm)
        else:
            # Save empty marker file if feature extraction failed
            np.savez_compressed(filepath, F=np.array([]), F_norm=np.array([]))


def load_feature_matrices(subj, gesture, inst):
    """
    Load all feature matrices for a gesture instance.
    
    Args:
        subj (str): Subject ID
        gesture (str): Gesture type
        inst (str): Instance name
    
    Returns:
        dict: Dictionary with 'full', 'sub_1', 'sub_2', 'sub_3' keys or None if not found
    """
    feature_matrices = {}
    matrix_types = ['full', 'sub_1', 'sub_2', 'sub_3']
    
    for matrix_type in matrix_types:
        filepath = get_feature_matrix_path(subj, gesture, inst, matrix_type)
        if not os.path.exists(filepath):
            return None
        
        data = np.load(filepath)
        F = data['F']
        F_norm = data['F_norm']
        
        if F.size == 0 or F_norm.size == 0:
            feature_matrices[matrix_type] = (None, None)
        else:
            feature_matrices[matrix_type] = (F, F_norm)
    
    return feature_matrices


# ==============================================================================
# PARALLEL DESCRIPTOR COMPUTATION (with resume support)
# ==============================================================================

def compute_instance_descriptor(args):
    """
    Process a single gesture instance and compute all descriptors in one pass.
    Also saves feature matrices to disk.
    
    Args:
        args (tuple): (subj, gesture, inst_name, inst_path)
    
    Returns:
        tuple: ((subj, gesture, inst_name), descriptor_dict) or None if failed.
    """
    subj, gesture, inst_name, inst_path = args
    print(f"[Descriptor] Processing Subject: {subj}, Gesture: {gesture}, Instance: {inst_name}")
    
    try:
        extractor = SilhouetteTunnelFeatureExtractor(bg_threshold=1500)
        result = extractor.compute_all_descriptors_from_gesture(inst_path)
        
        if result is None:
            print(f"[WARN] Failed to extract descriptors: {subj}/{gesture}/{inst_name}")
            return None
        
        # Extract feature matrices and descriptors
        feature_matrices = result.pop('feature_matrices')
        descriptor_dict = result  # Contains baseline, temporal, morph_cov, morph_temp
        
        # Save feature matrices to disk
        save_feature_matrices(subj, gesture, inst_name, feature_matrices)
        
        return ((subj, gesture, inst_name), descriptor_dict)
    
    except Exception as e:
        print(f"[ERROR] Exception processing {subj}/{gesture}/{inst_name}: {str(e)}")
        return None


def compute_all_descriptors(base_path, max_workers=4, force_reload=False):
    """
    Scan dataset and compute all descriptors in a single pass.
    Descriptors are cached and computation can be resumed if interrupted.
    
    Args:
        base_path (str): Root path of the dataset.
        max_workers (int): Number of parallel workers.
        force_reload (bool): If True, recompute all descriptors.
    
    Returns:
        dict: {(subj, gesture, inst): descriptor_dict}
    """
    cache_path = os.path.join(CACHE_DIR, "descriptors_cache.pkl")
    
    # Try loading cached descriptors
    if not force_reload and os.path.exists(cache_path):
        print(f"[INFO] Loading cached descriptors from {cache_path}")
        with open(cache_path, "rb") as f:
            descs = pickle.load(f)
        print(f"[INFO] Loaded {len(descs)} descriptor entries from cache.")
        return descs
    
    # Scan dataset structure
    print(f"\n[INFO] Scanning dataset: {base_path}")
    dataset_map = scan_dataset_structure(base_path)
    print(f"[INFO] Found {len(dataset_map)} gesture instances.")
    
    if len(dataset_map) == 0:
        print(f"[ERROR] No gesture instances found in {base_path}")
        return {}
    
    # Load progress to check what's already computed
    completed_keys = load_progress()
    print(f"[INFO] Found {len(completed_keys)} previously completed instances.")
    
    # Filter out already completed tasks
    if not force_reload:
        pending_tasks = [
            (subj, gesture, inst, inst_path)
            for (subj, gesture, inst), inst_path in dataset_map.items()
            if (subj, gesture, inst) not in completed_keys
        ]
    else:
        pending_tasks = [
            (subj, gesture, inst, inst_path)
            for (subj, gesture, inst), inst_path in dataset_map.items()
        ]
        completed_keys = set()  # Reset progress
    
    print(f"[INFO] {len(pending_tasks)} instances remaining to process.")
    
    # Load existing descriptors from cache if resuming
    descs = {}
    if not force_reload and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                descs = pickle.load(f)
            print(f"[INFO] Loaded {len(descs)} existing descriptors for resume.")
        except Exception as e:
            print(f"[WARN] Could not load existing cache: {e}")
            descs = {}
    
    if len(pending_tasks) == 0:
        print("[INFO] All instances already processed. Nothing to do.")
        return descs
    
    # Parallel descriptor computation
    print(f"\n[INFO] Computing descriptors for {len(pending_tasks)} instances with {max_workers} workers...")
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_instance_descriptor, task): task for task in pending_tasks}
        
        for f in as_completed(futures):
            result = f.result()
            if result:
                key, descriptor_dict = result
                descs[key] = descriptor_dict
                completed_keys.add(key)
                
                # Periodically save progress and cache
                if len(completed_keys) % 10 == 0:
                    save_progress(completed_keys)
                    with open(cache_path, "wb") as cache_f:
                        pickle.dump(descs, cache_f)
                    print(f"[INFO] Progress saved: {len(completed_keys)} instances completed.")
            
            completed += 1
            if completed % max(1, len(pending_tasks) // 10) == 0:
                print(f"[INFO] Progress: {completed}/{len(pending_tasks)} instances processed")
    
    print(f"[INFO] Descriptor computation complete. Processed {len(descs)}/{len(dataset_map)} instances.")
    
    # Final save
    print(f"[INFO] Caching {len(descs)} descriptors to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(descs, f)
    save_progress(completed_keys)
    print(f"[INFO] Cache and progress saved successfully.")
    
    return descs


# ==============================================================================
# DATA ORGANIZATION BY GESTURE
# ==============================================================================

def organize_by_gesture(descs):
    """
    Organize descriptors by gesture type.
    
    Args:
        descs (dict): All descriptors {(subj, gesture, inst): descriptor_dict}
    
    Returns:
        dict: {gesture_type: {(subj, gesture, inst): descriptor_dict}}
    """
    gesture_descs = {}
    for key, desc in descs.items():
        subj, gesture, inst = key
        if gesture not in gesture_descs:
            gesture_descs[gesture] = {}
        gesture_descs[gesture][key] = desc
    
    return gesture_descs


def organize_by_subject_and_instance(descs, gesture):
    """
    For a given gesture, organize descriptors by subject and instance number.
    
    Args:
        descs (dict): All descriptors
        gesture (str): Gesture type
    
    Returns:
        dict: {subj: [(inst_num, descriptor_dict), ...]} sorted by instance number
    """
    subject_instances = {}
    
    for (subj, gest, inst), desc in descs.items():
        if gest != gesture:
            continue
        
        if subj not in subject_instances:
            subject_instances[subj] = []
        
        # Extract instance number from inst folder name (e.g., "1", "2", etc.)
        try:
            inst_num = int(inst)
        except:
            inst_num = int(inst.split('_')[-1]) if '_' in inst else 0
        
        subject_instances[subj].append((inst_num, desc))
    
    # Sort by instance number
    for subj in subject_instances:
        subject_instances[subj].sort(key=lambda x: x[0])
    
    return subject_instances


# ==============================================================================
# SCORE COMPUTATION (Per-Gesture, with Genuine and Impostor Separation)
# ==============================================================================

def compute_descriptor_distance(desc1, desc2, mode, distance_metric='cosine'):
    """
    Compute distance between two descriptors.
    
    Args:
        desc1, desc2: Descriptors (either single array or list of arrays)
        mode (str): 'baseline', 'temporal', 'morph_cov', or 'morph_temp'
        distance_metric (str): 'cosine' or 'airm'
    
    Returns:
        float: Distance score
    """
    if distance_metric == 'airm':
        return compute_descriptor_distance_airm(desc1, desc2, mode)
    else:  # cosine (default)
        if mode in ["baseline", "temporal"]:
            return cosine(desc1, desc2)
        else:  # morph_cov or morph_temp (list of descriptors)
            if isinstance(desc1, list) and isinstance(desc2, list):
                return np.mean([cosine(d1, d2) for d1, d2 in zip(desc1, desc2)])
            else:
                return cosine(desc1, desc2)


def compute_scores_per_gesture(descs, gesture, mode, distance_metric='cosine'):
    """
    Compute genuine and impostor scores for a specific gesture and mode.
    
    Genuine Logic:
    - For each probe subject (09-16), split their instances in half
    - First 5 instances = enrollment templates
    - Last 5 instances = verification probes
    - Compare each probe against its templates
    
    Impostor Logic:
    - For each probe subject (09-16), compare against gallery subjects (01-08)
    - For a probe instance, compute distance to all gallery subject instances
    
    Args:
        descs (dict): All descriptors
        gesture (str): Gesture type to evaluate
        mode (str): Descriptor mode ('baseline', 'temporal', 'morph_cov', 'morph_temp')
        distance_metric (str): 'cosine' or 'airm'
    
    Returns:
        tuple: (genuine_scores, impostor_scores)
    """
    genuine_scores = []
    impostor_scores = []
    
    # Organize descriptors by subject and instance
    subject_instances = organize_by_subject_and_instance(descs, gesture)
    
    print(f"\n[INFO] Computing scores for gesture: {gesture}, mode: {mode}, metric: {distance_metric}")
    print(f"[DEBUG] Subjects found: {sorted(subject_instances.keys())}")
    
    # Gallery subjects: 01-08
    gallery_subjects = [f"{i:02d}" for i in range(1, 9)]
    # Probe subjects: 09-16
    probe_subjects = [f"{i:02d}" for i in range(9, 22)]
    
    # Verify we have gallery and probe data
    gallery_keys = [s for s in gallery_subjects if s in subject_instances]
    probe_keys = [s for s in probe_subjects if s in subject_instances]
    
    print(f"[DEBUG] Gallery subjects with data: {len(gallery_keys)}")
    print(f"[DEBUG] Probe subjects with data: {len(probe_keys)}")
    
    if not gallery_keys or not probe_keys:
        print(f"[WARN] Missing gallery or probe subjects for gesture {gesture}")
        return [], []
    
    # ========================================================================
    # GENUINE SCORES: Split probe subject data in half
    # ========================================================================
    for probe_subj in probe_keys:
        if probe_subj not in subject_instances:
            continue
        
        instances = subject_instances[probe_subj]
        
        if len(instances) < 2:
            continue
        
        # Split into enrollment (first half) and verification (second half)
        split_point = len(instances) // 2
        enrollment_instances = instances[:split_point]
        verification_instances = instances[split_point:]
        
        # Compare each verification instance against all enrollment instances
        for ver_inst_num, ver_desc in verification_instances:
            for enr_inst_num, enr_desc in enrollment_instances:
                distance = compute_descriptor_distance(
                    ver_desc[mode], enr_desc[mode], mode, distance_metric
                )
                genuine_scores.append(distance)
    
    # ========================================================================
    # IMPOSTOR SCORES: Probe subjects vs Gallery subjects (same gesture)
    # ========================================================================
    for probe_subj in probe_keys:
        if probe_subj not in subject_instances:
            continue
        
        probe_instances = subject_instances[probe_subj]
        
        # For each probe instance
        for probe_inst_num, probe_desc in probe_instances:
            # Compare against EACH gallery subject's EACH instance
            for gallery_subj in gallery_keys:
                if gallery_subj not in subject_instances:
                    continue
                
                gallery_instances = subject_instances[gallery_subj]
                
                # Collect ALL distances to all gallery instances
                for gallery_inst_num, gallery_desc in gallery_instances:
                    distance = compute_descriptor_distance(
                        probe_desc[mode], gallery_desc[mode], mode, distance_metric
                    )
                    impostor_scores.append(distance)
    
    print(f"[INFO] Gesture={gesture}, Mode={mode}, Metric={distance_metric}")
    print(f"[INFO] Genuine scores: {len(genuine_scores)}")
    print(f"[INFO] Impostor scores: {len(impostor_scores)}")
    
    return genuine_scores, impostor_scores


# ==============================================================================
# EER EVALUATION AND PLOTTING
# ==============================================================================

def evaluate_and_plot(genuine, impostor, title, histfile):
    """
    Compute and plot EER statistics.
    
    Args:
        genuine (list): Genuine match scores.
        impostor (list): Impostor match scores.
        title (str): Title for the evaluation.
        histfile (str): Output filename for the plot.
    """
    if len(genuine) == 0 or len(impostor) == 0:
        print(f"[ERROR] Empty score lists for {title}. Skipping.")
        return None

    # Convert distances to similarities (1 - distance)
    g_sim = 1.0 - np.array(genuine)
    i_sim = 1.0 - np.array(impostor)
    
    stats = get_eer_stats(g_sim, i_sim)
    print(f"[RESULT] {title}: EER = {stats.eer:.4f}")

    try:
        # Save current directory
        original_dir = os.getcwd()
        # Create results folder if it doesn't exist
        os.makedirs("./results", exist_ok=True)
        # Change to results folder
        os.chdir("./results")
        # Plot will be saved in results folder
        plot_eer_stats([stats], ids=[title])
        # Change back to original directory
        os.chdir(original_dir)
        print(f"[INFO] Plot saved to ./results/")
    except Exception as e:
        print(f"[WARN] Could not generate plot: {str(e)}")
        
    
    return stats


# ==============================================================================
# FEATURE MATRIX RETRIEVAL UTILITY
# ==============================================================================

def get_feature_matrices_for_instance(subj, gesture, inst):
    """
    Retrieve stored feature matrices for a specific instance.
    
    Args:
        subj (str): Subject ID
        gesture (str): Gesture type
        inst (str): Instance name
    
    Returns:
        dict: Dictionary with 'full', 'sub_1', 'sub_2', 'sub_3' keys
              Each value is a tuple (F_unnormalized, F_normalized)
              Returns None if not found
    """
    return load_feature_matrices(subj, gesture, inst)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Hand Gesture Authentication - Per-Gesture Ablation Study'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default="/Users/gandharvachhipa/Documents/ell715_assg4",
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers for feature extraction'
    )
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Force recomputation of all descriptors'
    )
    parser.add_argument(
        '--distance-metric',
        type=str,
        choices=['cosine', 'airm'],
        default='cosine',
        help='Distance metric to use: cosine (default) or airm (Riemannian)'
    )
    
    args = parser.parse_args()

    print("=" * 80)
    print("[INFO] Hand Gesture Authentication - Per-Gesture Ablation Study")
    print("[INFO] With Feature Matrix Storage and Resume Support")
    print(f"[INFO] Distance Metric: {args.distance_metric.upper()}")
    print("=" * 80)

    # Compute all descriptors in a single pass
    descs = compute_all_descriptors(args.dataset, args.workers, args.force_reload)

    if len(descs) == 0:
        print("[ERROR] No descriptors computed. Exiting.")
        exit(1)

    # Organize by gesture
    gesture_descs = organize_by_gesture(descs)
    print(f"\n[INFO] Found {len(gesture_descs)} unique gestures: {list(gesture_descs.keys())}")

    # Define ablation study experiments
    experiments = [
        ("baseline", "Baseline (Covariance Only)"),
        ("temporal", "Baseline + Temporal Hierarchy"),
        ("morph_cov", "Baseline + Morphology (Covariance)"),
        ("morph_temp", "Full System (Baseline + Temporal + Morphology)")
    ]

    # Store results for summary
    all_results = {}

    # Run ablation study PER GESTURE
    for gesture in sorted(gesture_descs.keys()):
        print("\n" + "=" * 80)
        print(f"[GESTURE] {gesture.upper()}")
        print("=" * 80)

        all_results[gesture] = {}

        for mode, name in experiments:
            print("\n" + "-" * 80)
            print(f"[RUNNING] {name} - {gesture}")
            print("-" * 80)

            genuine_scores, impostor_scores = compute_scores_per_gesture(
                gesture_descs[gesture], gesture, mode, args.distance_metric
            )
            
            if len(genuine_scores) > 0 and len(impostor_scores) > 0:
                # Sanitize filename for plot
                safe_gesture = gesture.replace(" ", "_").replace("/", "_")
                safe_mode = mode.replace(" ", "_")
                metric_suffix = f"_{args.distance_metric}" if args.distance_metric != 'cosine' else ""
                histfile = f"./results/{safe_gesture}_{safe_mode}{metric_suffix}_hist.png"
                os.makedirs("./results", exist_ok=True)
                
                stats = evaluate_and_plot(genuine_scores, impostor_scores,
                                         title=f"{name} - {gesture} ({args.distance_metric})",
                                         histfile=histfile)
                all_results[gesture][mode] = {
                    'eer': stats.eer if stats else None,
                    'num_genuine': len(genuine_scores),
                    'num_impostor': len(impostor_scores)
                }
            else:
                print(f"[ERROR] Insufficient scores. Genuine={len(genuine_scores)}, Impostor={len(impostor_scores)}")
                all_results[gesture][mode] = {
                    'eer': None,
                    'num_genuine': len(genuine_scores),
                    'num_impostor': len(impostor_scores)
                }

    # Print summary
    print("\n" + "=" * 80)
    print(f"[SUMMARY] Ablation Study Results - {args.distance_metric.upper()} Metric")
    print("=" * 80)
    
    for gesture in sorted(all_results.keys()):
        print(f"\n{gesture}:")
        for mode, name in experiments:
            if mode in all_results[gesture]:
                result = all_results[gesture][mode]
                if result['eer'] is not None:
                    eer_val = float(result['eer'])
                    print(f"  {name}: EER={eer_val:.4f}")
                    print(f"    Genuine={result['num_genuine']}, Impostor={result['num_impostor']}")
                else:
                    print(f"  {name}: FAILED (Genuine={result['num_genuine']}, Impostor={result['num_impostor']})")

    print("\n" + "=" * 80)
    print("[INFO] Ablation study complete.")
    print(f"[INFO] Distance metric used: {args.distance_metric.upper()}")
    print(f"[INFO] Descriptors stored in: {CACHE_DIR}")
    print(f"[INFO] Feature matrices stored in: {FEATURES_DIR}")
    print(f"[INFO] Results stored in: ./results/")
    print("=" * 80)
    
    # Example: How to retrieve feature matrices for a specific instance
    print("\n[INFO] Example: Retrieving feature matrices")
    print("[INFO] Use get_feature_matrices_for_instance(subj, gesture, inst)")
    print("[INFO] Example: matrices = get_feature_matrices_for_instance('01', 'wave', '1')")
    print("[INFO] Returns dict with keys: 'full', 'sub_1', 'sub_2', 'sub_3'")
    print("[INFO] Each value is tuple: (F_unnormalized, F_normalized)")


if __name__ == "__main__":
    main()