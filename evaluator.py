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


def compute_descriptor_distance(desc1, desc2, mode, distance_metric='cosine'):
    """
    Compute distance between two descriptors.
    
    FIXED: Consistent handling for both cosine and AIRM metrics.
    Both metrics now treat hierarchical descriptors the same way:
    - For 'baseline': Single covariance comparison
    - For 'temporal' and 'multiscale': Treat as single concatenated vector
    - For 'morph_cov' and 'morph_temp': Average distance across morphological parts
    
    Args:
        desc1, desc2: Descriptors (either single array or list of arrays)
        mode (str): 'baseline', 'temporal', 'morph_cov', 'morph_temp', or 'multiscale'
        distance_metric (str): 'cosine' or 'airm'
    
    Returns:
        float: Distance score
    """
    
    if mode == "baseline":
        # Both are single 105-dim covariance descriptors
        if distance_metric == 'cosine':
            return cosine(desc1, desc2)
        else:  # airm
            cov1 = reconstruct_covariance_from_descriptor(desc1)
            cov2 = reconstruct_covariance_from_descriptor(desc2)
            return compute_airm_distance(cov1, cov2)
    
    elif mode in ["temporal", "multiscale"]:
        # Both are concatenated temporal hierarchy descriptors
        # Treat as single high-dimensional vectors
        if distance_metric == 'cosine':
            return cosine(desc1, desc2)
        else:  # airm - not applicable for concatenated vectors
            # Fallback to cosine for concatenated multi-partition descriptors
            return cosine(desc1, desc2)
    
    elif mode == "morph_cov":
        # Both are lists of 4 covariance descriptors (full + 3 subs)
        if isinstance(desc1, list) and isinstance(desc2, list):
            distances = []
            for d1, d2 in zip(desc1, desc2):
                if distance_metric == 'cosine':
                    distances.append(cosine(d1, d2))
                else:  # airm
                    cov1 = reconstruct_covariance_from_descriptor(d1)
                    cov2 = reconstruct_covariance_from_descriptor(d2)
                    distances.append(compute_airm_distance(cov1, cov2))
            return np.mean(distances)
        else:
            # Fallback if not lists
            if distance_metric == 'cosine':
                return cosine(desc1, desc2)
            else:  # airm
                cov1 = reconstruct_covariance_from_descriptor(desc1)
                cov2 = reconstruct_covariance_from_descriptor(desc2)
                return compute_airm_distance(cov1, cov2)
    
    elif mode == "morph_temp":
        # Both are lists of 4 temporal hierarchy descriptors (full + 3 subs)
        # Each temporal descriptor is itself a concatenation of multiple partitions
        if isinstance(desc1, list) and isinstance(desc2, list):
            distances = []
            for d1, d2 in zip(desc1, desc2):
                # Treat each temporal descriptor as a single vector
                if distance_metric == 'cosine':
                    distances.append(cosine(d1, d2))
                else:  # airm - not applicable for concatenated vectors
                    distances.append(cosine(d1, d2))
            return np.mean(distances)
        else:
            # Fallback: treat as single temporal descriptors
            if distance_metric == 'cosine':
                return cosine(desc1, desc2)
            else:  # airm
                return cosine(desc1, desc2)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ==============================================================================
# DATASET SCANNING
# ==============================================================================

def scan_dataset_structure(base_path):
    """Scan the dataset directory structure without extracting features."""
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
    """Load processing progress from disk."""
    progress_file = get_progress_file()
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            return pickle.load(f)
    return set()


def save_progress(completed_keys):
    """Save processing progress to disk."""
    progress_file = get_progress_file()
    with open(progress_file, "wb") as f:
        pickle.dump(completed_keys, f)


def get_feature_matrix_path(subj, gesture, inst, matrix_type):
    """Get the file path for a specific feature matrix."""
    safe_gesture = gesture.replace(" ", "_").replace("/", "_")
    filename = f"{subj}_{safe_gesture}_{inst}_{matrix_type}.npz"
    return os.path.join(FEATURES_DIR, filename)


def save_feature_matrices(subj, gesture, inst, feature_matrices):
    """Save all feature matrices for a gesture instance."""
    for matrix_type, (F, F_norm) in feature_matrices.items():
        filepath = get_feature_matrix_path(subj, gesture, inst, matrix_type)
        if F is not None and F_norm is not None:
            np.savez_compressed(filepath, F=F, F_norm=F_norm)
        else:
            np.savez_compressed(filepath, F=np.array([]), F_norm=np.array([]))


def load_feature_matrices(subj, gesture, inst):
    """Load all feature matrices for a gesture instance."""
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
# PARALLEL DESCRIPTOR COMPUTATION
# ==============================================================================

def compute_instance_descriptor_baseline(args):
    """Process with baseline method (fixed threshold)."""
    subj, gesture, inst_name, inst_path, limit_frames = args
    print(f"[Descriptor] Processing Subject: {subj}, Gesture: {gesture}, Instance: {inst_name}")
    
    try:
        extractor = SilhouetteTunnelFeatureExtractor(bg_threshold=1500, use_otsu=False, limit_frames=limit_frames)
        result = extractor.compute_all_descriptors_from_gesture(inst_path, include_multiscale=False)
        
        if result is None:
            print(f"[WARN] Failed to extract descriptors: {subj}/{gesture}/{inst_name}")
            return None
        
        feature_matrices = result.pop('feature_matrices')
        save_feature_matrices(subj, gesture, inst_name, feature_matrices)
        
        return ((subj, gesture, inst_name), {
            'method': 'baseline',
            'descriptors': result
        })
    
    except Exception as e:
        print(f"[ERROR] Exception processing {subj}/{gesture}/{inst_name}: {str(e)}")
        return None
    
def compute_instance_descriptor_baseline_multiscale(args):
    """Process with baseline method (fixed threshold) including multiscale."""
    subj, gesture, inst_name, inst_path, limit_frames = args
    print(f"[Descriptor] Processing Subject: {subj}, Gesture: {gesture}, Instance: {inst_name}")
    
    try:
        extractor = SilhouetteTunnelFeatureExtractor(bg_threshold=1500, use_otsu=False, limit_frames=limit_frames)
        result = extractor.compute_all_descriptors_from_gesture(inst_path, include_multiscale=True)
        
        if result is None:
            print(f"[WARN] Failed to extract descriptors: {subj}/{gesture}/{inst_name}")
            return None
        
        feature_matrices = result.pop('feature_matrices')
        
        return ((subj, gesture, inst_name), {
            'method': 'baseline',
            'descriptors': result
        })
    
    except Exception as e:
        print(f"[ERROR] Exception processing {subj}/{gesture}/{inst_name}: {str(e)}")
        return None


def compute_instance_descriptor_otsu(args):
    """Process with Method 2: Otsu thresholding."""
    subj, gesture, inst_name, inst_path, limit_frames = args
    
    try:
        extractor = SilhouetteTunnelFeatureExtractor(bg_threshold=1500, use_otsu=True, limit_frames=limit_frames)
        result = extractor.compute_all_descriptors_from_gesture(inst_path, include_multiscale=True)
        
        if result is None:
            return None
        
        result.pop('feature_matrices')
        
        return ((subj, gesture, inst_name), {
            'method': 'otsu',
            'descriptors': result
        })
    
    except Exception as e:
        return None


def compute_all_descriptors(base_path, max_workers=8, force_reload=False, use_otsu=False, limit_frames=False):
    """
    Scan dataset and compute all descriptors.
    If use_otsu is True, compute both baseline and Otsu methods.
    
    Args:
        base_path (str): Root path of the dataset
        max_workers (int): Number of parallel workers
        force_reload (bool): Force recomputation
        use_otsu (bool): Whether to also compute Otsu variant
    
    Returns:
        dict: {(subj, gesture, inst): descriptor_dict}
    """
    cache_path = os.path.join(CACHE_DIR, "descriptors_cache.pkl")
    
    if not force_reload and os.path.exists(cache_path):
        print(f"[INFO] Loading cached descriptors from {cache_path}")
        with open(cache_path, "rb") as f:
            descs = pickle.load(f)
        print(f"[INFO] Loaded {len(descs)} descriptor entries from cache.")
        return descs
    
    print(f"\n[INFO] Scanning dataset: {base_path}")
    dataset_map = scan_dataset_structure(base_path)
    print(f"[INFO] Found {len(dataset_map)} gesture instances.")
    
    if len(dataset_map) == 0:
        print(f"[ERROR] No gesture instances found in {base_path}")
        return {}
    
    completed_keys = load_progress()
    print(f"[INFO] Found {len(completed_keys)} previously completed instances.")
    
    completed_keys = set()
    
    if not use_otsu:
        # --- Progress and Resume Logic (Only for non-Otsu mode) ---
        completed_keys = load_progress()
        print(f"[INFO] Found {len(completed_keys)} previously completed instances (Resume enabled).")
        
        if not force_reload:
            pending_tasks = [
                (subj, gesture, inst, inst_path, limit_frames)
                for (subj, gesture, inst), inst_path in dataset_map.items()
                if (subj, gesture, inst) not in completed_keys
            ]
        else:
            pending_tasks = [
                (subj, gesture, inst, inst_path, limit_frames)
                for (subj, gesture, inst), inst_path in dataset_map.items()
            ]
            completed_keys = set()
    
    else:
        # --- No Resume Logic (Otsu mode) ---
        print("[INFO] --use-otsu enabled: Resume/progress logic is disabled.")
        if not force_reload:
             print("[INFO] All instances will be processed.")
        
        pending_tasks = [
            (subj, gesture, inst, inst_path, limit_frames)
            for (subj, gesture, inst), inst_path in dataset_map.items()
        ]
        # We don't load or use completed_keys, so it remains an empty set.
    
    print(f"[INFO] {len(pending_tasks)} instances remaining to process.")
    
    descs = {}
    if not force_reload and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                descs = pickle.load(f)
            
            if not use_otsu:
                # If resuming, filter descriptors to only those we've completed
                descs = {k: v for k, v in descs.items() if k in completed_keys}
                print(f"[INFO] Loaded {len(descs)} existing descriptors for resume.")
            else:
                # In Otsu mode, we load the full cache and will merge/overwrite
                print(f"[INFO] Loaded {len(descs)} existing descriptors from cache.")

        except Exception as e:
            print(f"[WARN] Could not load existing cache: {e}")
            descs = {}
    
    if len(pending_tasks) == 0:
        print("[INFO] All instances already processed. Nothing to do.")
        return descs
    
    if use_otsu:
        print(f"\n[INFO] Computing descriptors for {len(pending_tasks)} instances with {max_workers} workers...")
        print("[INFO] Computing with Baseline and Method 2 (Otsu) variants...")
        
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit baseline tasks
            futures_baseline = {
                executor.submit(compute_instance_descriptor_baseline_multiscale, task): ('baseline', task) 
                for task in pending_tasks
            }
            
            # Submit Otsu tasks
            futures_otsu = {
                executor.submit(compute_instance_descriptor_otsu, task): ('otsu', task) 
                for task in pending_tasks
            }
            
            all_futures = {**futures_baseline, **futures_otsu}
            
            for f in as_completed(all_futures):
                result = f.result()
                if result:
                    key, result_dict = result
                    method = result_dict['method']
                    
                    # Merge results by key
                    if key not in descs:
                        descs[key] = {}
                    
                    descs[key][method] = result_dict['descriptors']
                    
                    # Only mark as completed when both methods are done
                    if 'baseline' in descs[key] and 'otsu' in descs[key]:
                        completed_keys.add(key)
                
                completed += 1
                if completed % max(1, len(all_futures) // 10) == 0:
                    print(f"[INFO] Progress: {completed}/{len(all_futures)} descriptor computations")
                
                # Periodically save
                if len(completed_keys) % 10 == 0 and len(completed_keys) > 0:
                    # save_progress(completed_keys) # <-- DISABLED IN OTSU
                    with open(cache_path, "wb") as cache_f:
                        pickle.dump(descs, cache_f)
    else:
        # Original behavior: only compute baseline
        print(f"\n[INFO] Computing descriptors for {len(pending_tasks)} instances with {max_workers} workers...")
        
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_instance_descriptor_baseline, task): task for task in pending_tasks}
            
            for f in as_completed(futures):
                result = f.result()
                if result:
                    key, result_dict = result
                    # For non-otsu mode, structure data differently for compatibility
                    descs[key] = result_dict['descriptors']
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
    if not use_otsu:
        save_progress(completed_keys)
        print(f"[INFO] Cache and progress saved successfully.")
    else:
        print(f"[INFO] Cache saved successfully (progress not saved in --use-otsu mode).")
    
    return descs


# ==============================================================================
# DATA ORGANIZATION BY GESTURE
# ==============================================================================

def organize_by_gesture(descs, use_otsu):
    """Organize descriptors by gesture type."""
    gesture_descs = {}
    for key, desc in descs.items():
        subj, gesture, inst = key
        if gesture not in gesture_descs:
            gesture_descs[gesture] = {}
        gesture_descs[gesture][key] = desc
    
    return gesture_descs


def organize_by_subject_and_instance(descs, gesture, use_otsu):
    """For a given gesture, organize descriptors by subject and instance number."""
    subject_instances = {}
    
    for (subj, gest, inst), desc in descs.items():
        if gest != gesture:
            continue
        
        if subj not in subject_instances:
            subject_instances[subj] = []
        
        try:
            inst_num = int(inst)
        except:
            inst_num = int(inst.split('_')[-1]) if '_' in inst else 0
        
        subject_instances[subj].append((inst_num, desc))
    
    for subj in subject_instances:
        subject_instances[subj].sort(key=lambda x: x[0])
    
    return subject_instances


# ==============================================================================
# SCORE COMPUTATION (Per-Gesture, with Genuine and Impostor Separation)
# ==============================================================================

def compute_scores_per_gesture(descs, gesture, mode, distance_metric='cosine', method='baseline', use_otsu=False):
    """
    Compute genuine and impostor scores for a specific gesture and mode.
    
    Args:
        descs (dict): All descriptors
        gesture (str): Gesture type to evaluate
        mode (str): Descriptor mode ('baseline', 'temporal', 'morph_cov', 'morph_temp', 'multiscale')
        distance_metric (str): 'cosine' or 'airm'
        method (str): Which method to use ('baseline', 'otsu') - only relevant if use_otsu is True
        use_otsu (bool): Whether Otsu variants are available
    
    Returns:
        tuple: (genuine_scores, impostor_scores)
    """
    genuine_scores = []
    impostor_scores = []
    
    subject_instances = organize_by_subject_and_instance(descs, gesture, use_otsu)
    
    print(f"\n[INFO] Computing scores for gesture: {gesture}, mode: {mode}, metric: {distance_metric}" + 
          (f", method: {method}" if use_otsu else ""))
    print(f"[DEBUG] Subjects found: {sorted(subject_instances.keys())}")
    
    gallery_subjects = [f"{i:02d}" for i in range(1, 9)]
    probe_subjects = [f"{i:02d}" for i in range(9, 22)]
    
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
        
        split_point = len(instances) // 2
        enrollment_instances = instances[:split_point]
        verification_instances = instances[split_point:]
        
        for ver_inst_num, ver_desc in verification_instances:
            for enr_inst_num, enr_desc in enrollment_instances:
                try:
                    if use_otsu:
                        ver_descriptor = ver_desc[method][mode]
                        enr_descriptor = enr_desc[method][mode]
                    else:
                        ver_descriptor = ver_desc[mode]
                        enr_descriptor = enr_desc[mode]
                    
                    if ver_descriptor is None or enr_descriptor is None:
                        continue
                    
                    distance = compute_descriptor_distance(
                        ver_descriptor, enr_descriptor, mode, distance_metric
                    )
                    genuine_scores.append(distance)
                except (KeyError, TypeError):
                    continue
    
    # ========================================================================
    # IMPOSTOR SCORES: Probe subjects vs Gallery subjects
    # ========================================================================
    for probe_subj in probe_keys:
        if probe_subj not in subject_instances:
            continue
        
        probe_instances = subject_instances[probe_subj]
        
        for probe_inst_num, probe_desc in probe_instances:
            for gallery_subj in gallery_keys:
                if gallery_subj not in subject_instances:
                    continue
                
                gallery_instances = subject_instances[gallery_subj]
                
                for gallery_inst_num, gallery_desc in gallery_instances:
                    try:
                        if use_otsu:
                            probe_descriptor = probe_desc[method][mode]
                            gallery_descriptor = gallery_desc[method][mode]
                        else:
                            probe_descriptor = probe_desc[mode]
                            gallery_descriptor = gallery_desc[mode]
                        
                        if probe_descriptor is None or gallery_descriptor is None:
                            continue
                        
                        distance = compute_descriptor_distance(
                            probe_descriptor, gallery_descriptor, mode, distance_metric
                        )
                        impostor_scores.append(distance)
                    except (KeyError, TypeError):
                        continue
    
    print(f"[INFO] Genuine scores: {len(genuine_scores)}")
    print(f"[INFO] Impostor scores: {len(impostor_scores)}")
    
    return genuine_scores, impostor_scores


# ==============================================================================
# EER EVALUATION AND PLOTTING
# ==============================================================================

def evaluate_and_plot(genuine, impostor, title, histfile):
    """
    Compute and plot EER statistics.
    
    FIXED: Proper distance-to-similarity conversion using negative distance.
    This works for both normalized (cosine) and unbounded (AIRM) distances.
    """
    if len(genuine) == 0 or len(impostor) == 0:
        print(f"[ERROR] Empty score lists for {title}. Skipping.")
        return None

    # Convert distance to similarity: sim = -dist
    # This correctly maps: low distance -> high similarity (genuine matches)
    #                     high distance -> low similarity (impostor matches)
    g_sim = -np.array(genuine)
    i_sim = -np.array(impostor)
    
    stats = get_eer_stats(g_sim, i_sim)
    print(f"[RESULT] {title}: EER = {stats.eer:.4f}")

    original_dir = os.getcwd()
    try:
        os.makedirs("./results", exist_ok=True)
        os.chdir("./results")
        plot_eer_stats([stats], ids=[title])
        print(f"[INFO] Plot saved to ./results/")
    except Exception as e:
        print(f"[WARN] Could not generate plot: {str(e)}")
    finally:
        os.chdir(original_dir)

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
    parser.add_argument(
        '--use-otsu',
        action='store_true',
        help='Enable Otsu thresholding variants (Method 2) for enhanced experiments'
    )
    parser.add_argument(
        '--limit-frames',
        action='store_true',
        help='Limit number of frames processed per gesture (for testing)'
    )
    
    args = parser.parse_args()

    print("=" * 80)
    if args.use_otsu:
        print("[INFO] Hand Gesture Authentication - Enhanced Ablation Study")
        print("[INFO] Baseline + Method 2 (Otsu) + Method 1 (Multi-Scale)")
        print(f"[INFO] Distance Metric: {args.distance_metric.upper()}")
        print("[INFO] 7 Total Experiments per Gesture")
    else:
        print("[INFO] Hand Gesture Authentication - Standard Ablation Study")
        print("[INFO] With Feature Matrix Storage and Resume Support")
        print(f"[INFO] Distance Metric: {args.distance_metric.upper()}")
        print("[INFO] 4 Experiments per Gesture")
    print("=" * 80)

    # Compute all descriptors
    descs = compute_all_descriptors(args.dataset, args.workers, args.force_reload, args.use_otsu, args.limit_frames)

    if len(descs) == 0:
        print("[ERROR] No descriptors computed. Exiting.")
        exit(1)

    # Organize by gesture
    gesture_descs = organize_by_gesture(descs, args.use_otsu)
    print(f"\n[INFO] Found {len(gesture_descs)} unique gestures: {list(gesture_descs.keys())}")

    # Define experiments based on whether Otsu is enabled
    if args.use_otsu:
        # Enhanced experiments with Otsu variants
        # FIXED: Renamed Experiments 6 and 7 to reflect temporal-only nature
        experiments = [
            # Experiments 1-4: Baseline (fixed threshold)
            ("baseline", "baseline", "Exp 1: Baseline (Covariance Only)"),
            ("baseline", "temporal", "Exp 2: Baseline + Temporal Hierarchy"),
            ("baseline", "morph_cov", "Exp 3: Baseline + Morphology (Covariance)"),
            ("baseline", "morph_temp", "Exp 4: Full System (Baseline + Temporal + Morphology)"),
            # Experiment 5: Method 2 (Otsu)
            ("otsu", "morph_temp", "Exp 5: Full System + Method 2 (Otsu Thresholding)"),
            # Experiment 6: Method 1 (Multi-Scale) - TEMPORAL ONLY
            ("baseline", "multiscale", "Exp 6: Temporal + Method 1 (Multi-Scale)"),
            # Experiment 7: Method 1 + Method 2 (Combined) - TEMPORAL ONLY
            ("otsu", "multiscale", "Exp 7: Temporal + Method 1&2 (Multi-Scale+Otsu)"),
        ]
    else:
        # Standard experiments (original 4 experiments)
        experiments = [
            ("baseline", "baseline", "Baseline (Covariance Only)"),
            ("baseline", "temporal", "Baseline + Temporal Hierarchy"),
            ("baseline", "morph_cov", "Baseline + Morphology (Covariance)"),
            ("baseline", "morph_temp", "Full System (Baseline + Temporal + Morphology)")
        ]

    # Store results for summary
    all_results = {}

    # Run ablation study PER GESTURE
    for gesture in sorted(gesture_descs.keys()):
        print("\n" + "=" * 80)
        print(f"[GESTURE] {gesture.upper()}")
        print("=" * 80)

        all_results[gesture] = {}

        for method, mode, name in experiments:
            print("\n" + "-" * 80)
            print(f"[RUNNING] {name} - {gesture}")
            print("-" * 80)

            genuine_scores, impostor_scores = compute_scores_per_gesture(
                gesture_descs[gesture], gesture, mode, args.distance_metric, method, args.use_otsu
            )
            
            if len(genuine_scores) > 0 and len(impostor_scores) > 0:
                # Sanitize filename for plot
                safe_gesture = gesture.replace(" ", "_").replace("/", "_")
                safe_mode = mode.replace(" ", "_")
                safe_name = name.replace(" ", "_").replace("+", "").replace(":", "").replace("&", "").replace("(", "").replace(")", "")
                metric_suffix = f"_{args.distance_metric}" if args.distance_metric != 'cosine' else ""
                histfile = f"./results/{safe_gesture}_{safe_name}{metric_suffix}_hist.png"
                os.makedirs("./results", exist_ok=True)
                
                stats = evaluate_and_plot(genuine_scores, impostor_scores,
                                         title=f"{name} - {gesture} ({args.distance_metric})",
                                         histfile=histfile)
                all_results[gesture][name] = {
                    'eer': stats.eer if stats else None,
                    'num_genuine': len(genuine_scores),
                    'num_impostor': len(impostor_scores)
                }
            else:
                print(f"[ERROR] Insufficient scores. Genuine={len(genuine_scores)}, Impostor={len(impostor_scores)}")
                all_results[gesture][name] = {
                    'eer': None,
                    'num_genuine': len(genuine_scores),
                    'num_impostor': len(impostor_scores)
                }

    # Print summary
    print("\n" + "=" * 80)
    if args.use_otsu:
        print(f"[SUMMARY] Enhanced Ablation Study Results - {args.distance_metric.upper()} Metric")
    else:
        print(f"[SUMMARY] Ablation Study Results - {args.distance_metric.upper()} Metric")
    print("=" * 80)
    
    for gesture in sorted(all_results.keys()):
        print(f"\n{gesture}:")
        for name, result in all_results[gesture].items():
            if result['eer'] is not None:
                eer_val = float(result['eer'])
                print(f"  {name}: EER={eer_val:.4f}")
                print(f"    Genuine={result['num_genuine']}, Impostor={result['num_impostor']}")
            else:
                print(f"  {name}: FAILED (Genuine={result['num_genuine']}, Impostor={result['num_impostor']})")

    print("\n" + "=" * 80)
    print("[INFO] Ablation study complete.")
    print(f"[INFO] Distance metric used: {args.distance_metric.upper()}")
    if args.use_otsu:
        print(f"[INFO] Enhanced mode: Otsu variants enabled")
        print(f"[INFO] NOTE: Experiments 6 & 7 use temporal descriptors only (no morphology)")
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
