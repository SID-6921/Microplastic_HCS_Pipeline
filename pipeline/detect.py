"""Cell detection module: find nuclei and measure fragmentation for apoptosis/necrosis phenotyping."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def detect_nuclei(
    dapi_channel: np.ndarray,
    pi_channel: np.ndarray | None = None,
    adaptive_block_size: int = 35,
    adaptive_c: float = -4.0,
    min_nucleus_area: int = 10,
    max_nucleus_area: int = 5000,
) -> tuple[np.ndarray, list[dict]]:
    """
    Detect cell nuclei from DAPI channel using adaptive thresholding.
    Measure nuclear fragmentation (apoptosis marker) and PI permeability (necrosis marker).
    
    Args:
        dapi_channel: DAPI fluorescence image (typically nucleus stain)
        pi_channel: PI fluorescence image (membrane permeability indicator), optional
        adaptive_block_size: Block size for adaptive thresholding
        adaptive_c: Constant subtracted from mean
        min_nucleus_area: Minimum nucleus size in pixels
        max_nucleus_area: Maximum nucleus size in pixels
    
    Returns:
        mask: Binary mask of detected nuclei
        nuclei: List of dicts with nucleus properties {area, centroid, pi_intensity, is_fragmented}
    """
    # Adaptive thresholding for nucleus detection
    thresh = cv2.adaptiveThreshold(
        dapi_channel,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        adaptive_c,
    )
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Label connected components
    labeled_array, num_features = ndimage.label(thresh)
    
    # Extract nucleus properties
    nuclei = []
    valid_mask = np.zeros_like(thresh)
    
    for nucleus_id in range(1, num_features + 1):
        nucleus_mask = labeled_array == nucleus_id
        area = np.sum(nucleus_mask)
        
        # Filter by size
        if area < min_nucleus_area or area > max_nucleus_area:
            continue
        
        # Calculate centroid
        y_coords, x_coords = np.where(nucleus_mask)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        
        # Measure PI intensity if available (necrosis/late apoptosis marker)
        pi_intensity = 0.0
        if pi_channel is not None:
            pi_intensity = float(np.mean(pi_channel[nucleus_mask]))
        
        # Measure nuclear fragmentation (apoptosis marker)
        # Count number of distinct sub-regions using more aggressive erosion
        eroded = cv2.erode(nucleus_mask.astype(np.uint8) * 255, kernel, iterations=2)
        eroded_labeled, eroded_count = ndimage.label(eroded)
        is_fragmented = eroded_count >= 2  # Multiple fragments = fragmented nucleus
        
        nuclei.append({
            "area": int(area),
            "centroid": (centroid_x, centroid_y),
            "pi_intensity": pi_intensity,
            "is_fragmented": bool(is_fragmented),
            "fragment_count": int(eroded_count),
        })
        
        valid_mask[nucleus_mask] = 1
    
    return valid_mask.astype(np.uint8) * 255, nuclei


def measure_cell_morphology(
    dapi_channel: np.ndarray,
    pi_channel: np.ndarray,
    nuclei: list[dict],
) -> dict:
    """
    Measure cell-level morphology indicators for apoptosis/necrosis classification.
    
    Returns dict with:
        - nuclear_fragmentation_index: fraction of fragmented nuclei
        - membrane_blebbing_score: boundary roughness (future: edge detection)
        - membrane_permeability_proxy: mean PI intensity (higher = necrosis)
        - chromatin_condensation_proxy: DAPI intensity variance
    """
    # Nuclear fragmentation index (0-1): fraction of fragmented nuclei
    if len(nuclei) > 0:
        fragmented_count = sum(1 for n in nuclei if n["is_fragmented"])
        nuclear_fragmentation_index = float(fragmented_count / len(nuclei))
    else:
        nuclear_fragmentation_index = 0.0
    
    # Membrane permeability proxy: mean PI intensity across all nuclei
    if len(nuclei) > 0:
        pi_intensities = [n["pi_intensity"] for n in nuclei if n["pi_intensity"] > 0]
        membrane_permeability_proxy = float(np.mean(pi_intensities)) if pi_intensities else 0.0
    else:
        membrane_permeability_proxy = 0.0
    
    # Chromatin condensation proxy: DAPI intensity variance (higher = condensed)
    chromatin_condensation_proxy = float(np.var(dapi_channel[dapi_channel > 0]))
    
    # Membrane blebbing score: placeholder (would require edge detection in future)
    membrane_blebbing_score = 0.0
    
    return {
        "nuclear_fragmentation_index": nuclear_fragmentation_index,
        "membrane_blebbing_score": membrane_blebbing_score,
        "membrane_permeability_proxy": membrane_permeability_proxy,
        "chromatin_condensation_proxy": chromatin_condensation_proxy,
    }
