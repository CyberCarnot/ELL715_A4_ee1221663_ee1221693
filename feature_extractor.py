import cv2
import numpy as np
import os
from numba import njit, prange
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# NUMBA-OPTIMIZED FUNCTIONS (Outside class for JIT compilation)
# ==============================================================================

@njit(parallel=True, cache=True)
def compute_directional_distances_numba(silhouette_tunnel):
    """
    Numba-optimized version of directional distance computation.
    Computes 10 directional Chebyshev distance transforms.
    
    Args:
        silhouette_tunnel (np.ndarray): TxHxW binary array
    
    Returns:
        tuple: 10 distance maps as separate arrays
    """
    T, H, W = silhouette_tunnel.shape
    
    # Initialize all distance maps
    d_N = np.zeros((T, H, W), dtype=np.float32)
    d_S = np.zeros((T, H, W), dtype=np.float32)
    d_W = np.zeros((T, H, W), dtype=np.float32)
    d_E = np.zeros((T, H, W), dtype=np.float32)
    d_NW = np.zeros((T, H, W), dtype=np.float32)
    d_NE = np.zeros((T, H, W), dtype=np.float32)
    d_SW = np.zeros((T, H, W), dtype=np.float32)
    d_SE = np.zeros((T, H, W), dtype=np.float32)
    d_f = np.zeros((T, H, W), dtype=np.float32)
    d_b = np.zeros((T, H, W), dtype=np.float32)
    
    # Compute spatial directions for each frame
    for t in prange(T):
        mask = silhouette_tunnel[t].astype(np.float32)
        
        # South (top to bottom)
        for i in range(1, H):
            for j in range(W):
                d_S[t, i, j] = (d_S[t, i-1, j] + 1) * mask[i, j]
        
        # North (bottom to top)
        for i in range(H - 2, -1, -1):
            for j in range(W):
                d_N[t, i, j] = (d_N[t, i+1, j] + 1) * mask[i, j]
        
        # East (left to right)
        for i in range(H):
            for j in range(1, W):
                d_E[t, i, j] = (d_E[t, i, j-1] + 1) * mask[i, j]
        
        # West (right to left)
        for i in range(H):
            for j in range(W - 2, -1, -1):
                d_W[t, i, j] = (d_W[t, i, j+1] + 1) * mask[i, j]
        
        # Southeast (diagonal)
        for i in range(1, H):
            for j in range(1, W):
                d_SE[t, i, j] = (d_SE[t, i-1, j-1] + 1) * mask[i, j]
        
        # Northwest (diagonal)
        for i in range(H - 2, -1, -1):
            for j in range(W - 2, -1, -1):
                d_NW[t, i, j] = (d_NW[t, i+1, j+1] + 1) * mask[i, j]
        
        # Southwest (diagonal)
        for i in range(1, H):
            for j in range(W - 2, -1, -1):
                d_SW[t, i, j] = (d_SW[t, i-1, j+1] + 1) * mask[i, j]
        
        # Northeast (diagonal)
        for i in range(H - 2, -1, -1):
            for j in range(1, W):
                d_NE[t, i, j] = (d_NE[t, i+1, j-1] + 1) * mask[i, j]
    
    # Forward temporal (time forward)
    for t in range(1, T):
        for i in range(H):
            for j in range(W):
                d_f[t, i, j] = (d_f[t-1, i, j] + 1) * silhouette_tunnel[t, i, j]
    
    # Backward temporal (time backward)
    for t in range(T - 2, -1, -1):
        for i in range(H):
            for j in range(W):
                d_b[t, i, j] = (d_b[t+1, i, j] + 1) * silhouette_tunnel[t, i, j]
    
    return d_E, d_W, d_N, d_S, d_NE, d_NW, d_SE, d_SW, d_f, d_b


@njit(cache=True)
def extract_feature_vectors_numba(silhouette_tunnel, original_depth_volume, 
                                   d_E, d_W, d_N, d_S, d_NE, d_NW, d_SE, d_SW, d_f, d_b):
    """
    Numba-optimized feature vector extraction.
    
    Args:
        silhouette_tunnel (np.ndarray): TxHxW binary mask
        original_depth_volume (np.ndarray): TxHxW depth values
        d_* (np.ndarray): Distance maps
    
    Returns:
        np.ndarray: 14xN feature matrix
    """
    T, H, W = silhouette_tunnel.shape
    
    # Count foreground pixels
    N = 0
    for t in range(T):
        for i in range(H):
            for j in range(W):
                if silhouette_tunnel[t, i, j] > 0:
                    N += 1
    
    if N == 0:
        return np.zeros((14, 0), dtype=np.float32)
    
    # Allocate feature matrix
    F = np.zeros((14, N), dtype=np.float32)
    
    # Extract features
    idx = 0
    for t in range(T):
        for y in range(H):
            for x in range(W):
                if silhouette_tunnel[t, y, x] > 0:
                    F[0, idx] = x
                    F[1, idx] = y
                    F[2, idx] = t
                    F[3, idx] = original_depth_volume[t, y, x]
                    F[4, idx] = d_E[t, y, x]
                    F[5, idx] = d_W[t, y, x]
                    F[6, idx] = d_N[t, y, x]
                    F[7, idx] = d_S[t, y, x]
                    F[8, idx] = d_NE[t, y, x]
                    F[9, idx] = d_NW[t, y, x]
                    F[10, idx] = d_SE[t, y, x]
                    F[11, idx] = d_SW[t, y, x]
                    F[12, idx] = d_f[t, y, x]
                    F[13, idx] = d_b[t, y, x]
                    idx += 1
    
    return F


@njit(parallel=True, cache=True)
def normalize_features_numba(F, norm_params):
    """
    Numba-optimized row-wise min-max normalization using provided parameters.
    
    Args:
        F (np.ndarray): 14xN feature matrix
        norm_params (np.ndarray): 14x2 array with [min, max] for each feature
    
    Returns:
        np.ndarray: Normalized 14xN feature matrix
    """
    num_features, N = F.shape
    F_normalized = np.zeros((num_features, N), dtype=np.float32)
    
    for i in prange(num_features):
        row = F[i, :]
        min_val = norm_params[i, 0]
        max_val = norm_params[i, 1]
        denominator = max_val - min_val
        
        if denominator > 0:
            for j in range(N):
                F_normalized[i, j] = (row[j] - min_val) / denominator
        else:
            # If all values are the same, normalized to 0
            for j in range(N):
                F_normalized[i, j] = 0.0
    
    return F_normalized


@njit(cache=True)
def compute_covariance_numba(F):
    """
    Numba-optimized covariance computation with upper triangle extraction.
    
    Args:
        F (np.ndarray): 14xN feature matrix
    
    Returns:
        np.ndarray: 105-dim upper triangle of covariance matrix
    """
    num_features, N = F.shape
    
    if N == 0:
        return np.zeros(105, dtype=np.float32)
    
    # Compute mean for each feature
    means = np.zeros(num_features, dtype=np.float32)
    for i in range(num_features):
        means[i] = np.mean(F[i, :])
    
    # Center the data
    F_centered = np.zeros((num_features, N), dtype=np.float32)
    for i in range(num_features):
        for j in range(N):
            F_centered[i, j] = F[i, j] - means[i]
    
    # Compute covariance matrix
    cov = np.zeros((num_features, num_features), dtype=np.float32)
    for i in range(num_features):
        for j in range(num_features):
            sum_val = 0.0
            for k in range(N):
                sum_val += F_centered[i, k] * F_centered[j, k]
            cov[i, j] = sum_val / N
    
    # Extract upper triangle
    descriptor = np.zeros(105, dtype=np.float32)
    idx = 0
    for i in range(num_features):
        for j in range(i, num_features):
            descriptor[idx] = cov[i, j]
            idx += 1
    
    return descriptor


# ==============================================================================
# MAIN CLASS (Uses optimized functions)
# ==============================================================================

class SilhouetteTunnelFeatureExtractor:
    """
    Implements Module 1 of the assignment: Silhouette Tunnel and Feature Extraction.
    
    Optimized with Numba JIT compilation for fast processing.
    Supports Method 1 (Multi-Scale) and Method 2 (Otsu Thresholding) via parameters.
    """

    def __init__(self, bg_threshold=10, use_otsu=False, limit_frames=False):
        """
        Initializes the extractor with configuration parameters.

        Args:
            bg_threshold (int): The threshold for background subtraction (fixed threshold mode).
            use_otsu (bool): If True, use adaptive Otsu thresholding instead of fixed threshold.
        """
        self.bg_threshold = bg_threshold
        self.use_otsu = use_otsu
        self.limit_frames = limit_frames

    def _load_frame(self, folder_path, frame_num):
        """
        Loads and preprocesses a single 16-bit depth frame from lsb and msb files.
        """
        filepath_lsb = os.path.join(folder_path, 'lsb', f'lsb{frame_num}.png')
        filepath_msb = os.path.join(folder_path, 'msb', f'msb{frame_num}.png')

        if not (os.path.exists(filepath_lsb) and os.path.exists(filepath_msb)):
            return None

        img_lsb = cv2.imread(filepath_lsb, cv2.IMREAD_GRAYSCALE)
        img_msb = cv2.imread(filepath_msb, cv2.IMREAD_GRAYSCALE)

        depth_frame = (img_msb.astype(np.uint16) << 8) | img_lsb.astype(np.uint16)

        return depth_frame

    def _load_sequence(self, folder_path):
        """
        Loads frames for a gesture instance.
        
        If limit_frames is True, loading is constrained based on gesture type.
        Otherwise, all available gesture frames are loaded.

        Frame constraints by gesture (if limit_frames=True):
        - Compass: 119 frames
        - Piano: 79 frames
        - Push: 60 frames
        - UCDO: 79 frames
        
        Args:
            folder_path (str): Path to gesture instance directory.
            limit_frames (bool): Whether to limit frames based on gesture type.
            
        Returns:
            tuple: (gesture_frames, background_frame)
            
        Raises:
            ValueError: If the directory is not found, has less than 2 frames,
                        or if no gesture frames could be loaded.
        """
        
        # Get total number of frames available in the directory
        try:
            lsb_path = os.path.join(folder_path, 'lsb')
            num_frames_available = len(os.listdir(lsb_path))
        except FileNotFoundError:
            raise ValueError(f"Directory not found: {lsb_path}")
        except Exception as e:
            raise ValueError(f"Error reading directory {lsb_path}: {e}")

        if num_frames_available < 2:
            raise ValueError(f"Not enough frames in {folder_path}. Found {num_frames_available}, need at least 2 (1 background, 1+ gesture).")

        # Frame 1 is always the background
        background_frame = self._load_frame(folder_path, 1)
        
        num_gesture_frames_to_load = 0
        available_gesture_frames = num_frames_available - 1 # All frames except background

        if self.limit_frames:
            # --- Logic from Version 1 ---
            # Define frame limits (number of gesture frames) by gesture type
            gesture_frame_limits = {
                'Compass': 119,
                'Piano': 79,
                'Push': 60,
                'UCDO': 79
            }
            
            # Extract gesture type from folder path
            gesture_name = None
            for gest in gesture_frame_limits.keys():
                if gest in folder_path:
                    gesture_name = gest
                    break
            
            # Default to 51 gesture frames if gesture not recognized
            max_gesture_frames_limit = gesture_frame_limits.get(gesture_name, 51)
            
            # Load the minimum of the gesture limit or the available frames
            num_gesture_frames_to_load = min(max_gesture_frames_limit, available_gesture_frames)

        else:
            # --- Logic from Version 2 ---
            # Load all available gesture frames
            num_gesture_frames_to_load = available_gesture_frames

        # Load gesture frames starting from frame 2
        gesture_frames = []
        # Loop from 2 up to (2 + num_gesture_frames_to_load)
        # e.g., if loading 60 frames, range(2, 62) loads indices 2..61
        for i in range(2, 2 + num_gesture_frames_to_load):
            frame = self._load_frame(folder_path, i)
            if frame is not None:
                gesture_frames.append(frame)
        
        if not gesture_frames:
            # Safety check from both versions
            raise ValueError(f"No gesture frames were loaded from {folder_path}")
            
        return gesture_frames, background_frame

    def _get_clean_silhouettes(self, gesture_frames, background_frame):
        """
        Performs background subtraction and largest connected component analysis.
        
        Supports both fixed threshold (baseline) and adaptive Otsu thresholding (Method 2).
        """
        clean_masks = []
        for frame in gesture_frames:
            diff = cv2.absdiff(background_frame, frame)
            
            # Choose thresholding method
            if self.use_otsu:
                # Method 2: Otsu Thresholding
                _, fg_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Baseline: Fixed threshold
                _, fg_mask = cv2.threshold(diff, self.bg_threshold, 255, cv2.THRESH_BINARY)
            
            fg_mask = fg_mask.astype(np.uint8)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, 8, cv2.CV_32S)
            
            clean_mask = np.zeros_like(fg_mask)
            if num_labels > 1:
                largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                clean_mask[labels == largest_component_label] = 1
            
            clean_masks.append(clean_mask)
            
        return np.array(clean_masks, dtype=np.uint8)

    def _compute_directional_distances(self, silhouette_tunnel):
        """
        Wrapper for Numba-optimized directional distance computation.
        """
        return compute_directional_distances_numba(silhouette_tunnel)

    def _compute_normalization_params(self, F):
        """
        Compute min and max for each feature dimension.
        
        Args:
            F (np.ndarray): 14xN feature matrix
        
        Returns:
            np.ndarray: 14x2 array with [min, max] for each feature
        """
        num_features = F.shape[0]
        norm_params = np.zeros((num_features, 2), dtype=np.float32)
        
        for i in range(num_features):
            norm_params[i, 0] = np.min(F[i, :])
            norm_params[i, 1] = np.max(F[i, :])
        
        return norm_params

    def _normalize_features(self, F, norm_params):
        """
        Wrapper for Numba-optimized normalization using provided parameters.
        """
        return normalize_features_numba(F, norm_params)
    
    def _feature_from_tunnel(self, silhouette_tunnel, original_depth_volume, norm_params=None):
        """
        Extracts 14D feature vectors from the silhouette tunnel (optimized version).
        
        Args:
            silhouette_tunnel (np.ndarray): TxHxW binary silhouette mask.
            original_depth_volume (np.ndarray): TxHxW original depth values.
            norm_params (np.ndarray): Optional normalization parameters from full hand
        
        Returns:
            tuple: (F_unnormalized, F_normalized, silhouette_tunnel, original_depth_volume)
        """
        # Compute distance maps using optimized function
        d_E, d_W, d_N, d_S, d_NE, d_NW, d_SE, d_SW, d_f, d_b = self._compute_directional_distances(silhouette_tunnel)
        
        # Extract feature vectors using optimized function
        F = extract_feature_vectors_numba(
            silhouette_tunnel, original_depth_volume,
            d_E, d_W, d_N, d_S, d_NE, d_NW, d_SE, d_SW, d_f, d_b
        )
        
        if F.shape[1] == 0:
            return None, None, None, None
        
        # Normalize using provided parameters or compute new ones
        if norm_params is None:
            norm_params = self._compute_normalization_params(F)
        
        F_normalized = self._normalize_features(F, norm_params)
        
        return F, F_normalized, silhouette_tunnel, original_depth_volume

    def process_gesture(self, folder_path):
        """
        Main method to process a gesture and extract feature matrices.

        Args:
            folder_path (str): Path to the gesture instance directory.

        Returns:
            tuple: (F_unnormalized, F_normalized, silhouette_tunnel, original_depth_volume)
                   Returns (None, None, None, None) if processing fails.
        """
        try:
            gesture_frames, background_frame = self._load_sequence(folder_path)
            original_depth_volume = np.array(gesture_frames)
            silhouette_tunnel = self._get_clean_silhouettes(gesture_frames, background_frame)
            if np.sum(silhouette_tunnel) == 0: 
                return None, None, None, None
            
            return self._feature_from_tunnel(silhouette_tunnel, original_depth_volume)

        except Exception as e:
            print(f"[ERROR] Error processing gesture: {str(e)}")
            return None, None, None, None
        
    def segment_hand_by_depth(self, original_depth_volume, full_silhouette_tunnel, k=3, min_component_size=20):
        """
        Segments the hand silhouette into K sub-silhouettes based on depth.

        Args:
            original_depth_volume (np.ndarray): T x H x W array of original depth values.
            full_silhouette_tunnel (np.ndarray): T x H x W array of binary hand masks.
            k (int): The number of depth partitions to create (paper uses 3).
            min_component_size (int): Remove components smaller than this size (paper uses 20).

        Returns:
            list: A list of K sub-silhouette tunnels (T x H x W numpy arrays).
        """
        T, H, W = original_depth_volume.shape
        sub_tunnels = [[] for _ in range(k)]

        for t in range(T):
            frame_mask = full_silhouette_tunnel[t]
            depth_frame = original_depth_volume[t]
            
            hand_depth_values = depth_frame[frame_mask > 0]

            if len(hand_depth_values) < k:
                for i in range(k):
                    sub_tunnels[i].append(np.zeros_like(frame_mask))
                continue

            percentiles = np.linspace(0, 100, k + 1)
            thresholds = np.percentile(hand_depth_values, percentiles)
            
            for i in range(k):
                lower_bound = thresholds[i]
                upper_bound = thresholds[i+1]
                
                if i == k - 1:
                    sub_mask = (depth_frame >= lower_bound) & (depth_frame <= upper_bound)
                else:
                    sub_mask = (depth_frame >= lower_bound) & (depth_frame < upper_bound)
                
                sub_mask = sub_mask & (frame_mask > 0)
                sub_mask = sub_mask.astype(np.uint8)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sub_mask, 8)
                cleaned_mask = np.zeros_like(sub_mask)
                if num_labels > 1:
                    for label_idx in range(1, num_labels):
                        if stats[label_idx, cv2.CC_STAT_AREA] >= min_component_size:
                            cleaned_mask[labels == label_idx] = 1
                
                sub_tunnels[i].append(cleaned_mask)

        return [np.array(tunnel) for tunnel in sub_tunnels]

    def compute_multiscale_descriptor(self, folder_path, baseline_temp, scales=[0.5, 0.25]):
        """
        Method 1: Multi-Scale Silhouette Tunnels (OPTIMIZED)
        
        Computes descriptors at multiple spatial scales and concatenates them.
        OPTIMIZATION: Reuses baseline_temp (scale=1.0) to avoid recomputation.
        
        Args:
            folder_path (str): Path to the gesture instance directory.
            baseline_temp (np.ndarray): Pre-computed baseline temporal descriptor (735-dim)
            scales (list): Spatial scales to process (default: [0.5, 0.25], skip 1.0)
        
        Returns:
            np.ndarray: Multi-scale concatenated descriptor (2205-dim: 735*3)
        """
        try:
            gesture_frames, background_frame = self._load_sequence(folder_path)
            original_depth_volume = np.array(gesture_frames)
            silhouette_tunnel = self._get_clean_silhouettes(gesture_frames, background_frame)
            
            if np.sum(silhouette_tunnel) == 0:
                return np.zeros(2205, dtype=np.float32)
            
            T, H, W = silhouette_tunnel.shape
            
            # Compute normalization parameters from full resolution (scale 1.0)
            d_E, d_W, d_N, d_S, d_NE, d_NW, d_SE, d_SW, d_f, d_b = self._compute_directional_distances(silhouette_tunnel)
            F_full = extract_feature_vectors_numba(
                silhouette_tunnel, original_depth_volume,
                d_E, d_W, d_N, d_S, d_NE, d_NW, d_SE, d_SW, d_f, d_b
            )
            
            if F_full.shape[1] == 0:
                return np.zeros(2205, dtype=np.float32)
            
            norm_params = self._compute_normalization_params(F_full)
            
            # Start with baseline (scale 1.0) - already computed!
            scale_descriptors = [baseline_temp]
            
            # Only compute scales [0.5, 0.25]
            for scale in scales:
                scaled_h, scaled_w = int(H * scale), int(W * scale)
                if scaled_h < 1 or scaled_w < 1:
                    scale_descriptors.append(np.zeros(735, dtype=np.float32))
                    continue
                
                scaled_tunnel = np.zeros((T, scaled_h, scaled_w), dtype=np.uint8)
                scaled_depth = np.zeros((T, scaled_h, scaled_w), dtype=np.float32)
                
                for t in range(T):
                    scaled_tunnel[t] = cv2.resize(
                        silhouette_tunnel[t].astype(np.uint8),
                        (scaled_w, scaled_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    scaled_depth[t] = cv2.resize(
                        original_depth_volume[t].astype(np.float32),
                        (scaled_w, scaled_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Extract features using SAME normalization params as full scale
                F, F_normalized, _, _ = self._feature_from_tunnel(scaled_tunnel, scaled_depth, norm_params)
                
                if F is not None and F_normalized is not None:
                    scale_desc = build_temporal_hierarchy_descriptor(F_normalized)
                    scale_descriptors.append(scale_desc)
                else:
                    scale_descriptors.append(np.zeros(735, dtype=np.float32))

            # Concatenate: baseline (735) + scale_0.5 (735) + scale_0.25 (735) = 2205-dim
            concat_descriptor = np.concatenate(scale_descriptors).astype(np.float32)
            
            return concat_descriptor
    
        except Exception as e:
            print(f"[ERROR] Exception in compute_multiscale_descriptor: {str(e)}")
            return np.zeros(2205, dtype=np.float32)

    def compute_all_descriptors_from_gesture(self, folder_path, include_multiscale=False):
        """
        Process gesture and compute all descriptors in one pass (optimized).
        
        This method:
        1. Loads and processes the gesture to extract F_norm (full feature matrix)
        2. Computes baseline descriptors (covariance and temporal hierarchy)
        3. Segments hand into 3 sub-silhouettes by depth
        4. Computes morphology descriptors for each sub-silhouette (using SAME normalization)
        5. Optionally computes multi-scale descriptor (Method 1) if include_multiscale=True
        6. Returns all descriptors AND feature matrices in a single dictionary
        
        Args:
            folder_path (str): Path to the gesture instance directory.
            include_multiscale (bool): If True, compute multi-scale descriptor (Method 1).
        
        Returns:
            dict: {
                'baseline': 105-dim covariance descriptor,
                'temporal': 735-dim temporal hierarchy descriptor,
                'morph_cov': [4 x 105-dim] (full + 3 sub covariance descriptors),
                'morph_temp': [4 x 735-dim] (full + 3 sub temporal hierarchy descriptors),
                'multiscale': 2205-dim multi-scale descriptor (only if include_multiscale=True),
                'feature_matrices': {
                    'full': (F_unnormalized, F_normalized),
                    'sub_1': (F_unnormalized, F_normalized),
                    'sub_2': (F_unnormalized, F_normalized),
                    'sub_3': (F_unnormalized, F_normalized)
                }
            }
            Returns None if processing fails.
        """
        try:
            # Step 1: Load and process gesture
            gesture_frames, background_frame = self._load_sequence(folder_path)
            original_depth_volume = np.array(gesture_frames)
            silhouette_tunnel = self._get_clean_silhouettes(gesture_frames, background_frame)
            
            if np.sum(silhouette_tunnel) == 0:
                return None
            
            # Step 2: Extract full feature matrix and compute normalization params
            F, F_normalized, _, _ = self._feature_from_tunnel(silhouette_tunnel, original_depth_volume)
            
            if F is None or F_normalized is None:
                return None
            
            # Compute normalization parameters from FULL hand
            norm_params = self._compute_normalization_params(F)
            
            # Initialize feature matrices storage
            feature_matrices = {
                'full': (F, F_normalized)
            }
            
            # Step 3: Compute baseline descriptors (using optimized covariance)
            baseline_cov = compute_covariance_descriptor(F_normalized)
            baseline_temp = build_temporal_hierarchy_descriptor(F_normalized)
            
            # Step 4: Segment hand into 3 sub-silhouettes by depth
            sub_tunnels = self.segment_hand_by_depth(original_depth_volume, silhouette_tunnel, k=3)
            
            # Step 5: Compute morphology descriptors using SAME normalization params
            morph_cov = [baseline_cov]  # Start with full silhouette covariance
            morph_temp = [baseline_temp]  # Start with full silhouette temporal
            
            for idx, sub_tunnel in enumerate(sub_tunnels, 1):
                # Extract features using SAME normalization params as full hand
                sub_F, sub_F_normalized, _, _ = self._feature_from_tunnel(sub_tunnel, original_depth_volume, norm_params)
                
                if sub_F is not None and sub_F_normalized is not None:
                    # Store sub-silhouette feature matrices
                    feature_matrices[f'sub_{idx}'] = (sub_F, sub_F_normalized)
                    sub_cov = compute_covariance_descriptor(sub_F_normalized)
                    morph_cov.append(sub_cov)
                    
                    # Compute temporal hierarchy descriptor for sub-silhouette
                    sub_temp = build_temporal_hierarchy_descriptor(sub_F_normalized)
                    morph_temp.append(sub_temp)
                else:
                    # If extraction fails for sub-silhouette, append zeros
                    feature_matrices[f'sub_{idx}'] = (None, None)
                    morph_cov.append(np.zeros(105, dtype=np.float32))
                    morph_temp.append(np.zeros(735, dtype=np.float32))
            
            # Build result dictionary
            result = {
                'baseline': baseline_cov,
                'temporal': baseline_temp,
                'morph_cov': morph_cov,
                'morph_temp': morph_temp,
                'feature_matrices': feature_matrices
            }
            
            # Step 6: Optionally compute Multi-Scale descriptor (Method 1)
            if include_multiscale:
                multiscale_desc = self.compute_multiscale_descriptor(folder_path, baseline_temp)
                result['multiscale'] = multiscale_desc
            
            return result
        
        except Exception as e:
            print(f"[ERROR] Exception in compute_all_descriptors_from_gesture: {str(e)}")
            return None


def compute_covariance_descriptor(feature_matrix):
    """
    Computes the covariance descriptor from the 14xN feature matrix (optimized wrapper).

    Args:
        feature_matrix (np.ndarray): A 14xN NumPy array.

    Returns:
        np.ndarray: A 105-dimensional feature vector (upper triangle of covariance matrix).
    """
    if feature_matrix is None or feature_matrix.shape[1] == 0:
        return np.zeros(105, dtype=np.float32)
    
    return compute_covariance_numba(feature_matrix)


def build_temporal_hierarchy_descriptor(normalized_feature_matrix, num_levels=3):
    """
    Builds a hierarchical descriptor from a PRE-NORMALIZED feature matrix.
    Uses optimized covariance computation.

    This descriptor divides the temporal sequence into increasingly finer partitions
    and computes covariance descriptors for each partition. The final descriptor is
    a concatenation of all partition descriptors.

    Args:
        normalized_feature_matrix (np.ndarray): The 14xN normalized feature matrix.
        num_levels (int): The number of levels in the hierarchy (default: 3).

    Returns:
        np.ndarray: A concatenated feature vector of length 735 (for num_levels=3).
                    Structure: [Level1_covariances | Level2_covariances | Level3_covariances]
    """
    if normalized_feature_matrix is None or normalized_feature_matrix.shape[1] == 0:
        total_descriptors = sum([2**(i-1) for i in range(1, num_levels + 1)])
        return np.zeros(total_descriptors * 105, dtype=np.float32)

    # Extract time coordinates (3rd feature is normalized time index)
    time_coordinates = normalized_feature_matrix[2, :]
    
    all_descriptors = []
    
    # Process each level of the hierarchy
    for level in range(1, num_levels + 1):
        num_partitions = 2**(level - 1)
        
        # Process each partition at this level
        for i in range(num_partitions):
            start_time = i / num_partitions
            end_time = (i + 1) / num_partitions
            
            # Create temporal mask for this partition
            if i == num_partitions - 1:
                mask = (time_coordinates >= start_time) & (time_coordinates <= end_time)
            else:
                mask = (time_coordinates >= start_time) & (time_coordinates < end_time)
            
            # Extract features for this temporal partition
            partition_feature_matrix = normalized_feature_matrix[:, mask]
            
            # Compute covariance descriptor for this partition (optimized!)
            partition_descriptor = compute_covariance_descriptor(partition_feature_matrix)
            all_descriptors.append(partition_descriptor)

    # Concatenate all descriptors from all levels
    return np.concatenate(all_descriptors).astype(np.float32)
