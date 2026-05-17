"""Feature extraction: 18 morphological descriptors for cell death phenotyping."""
from __future__ import annotations

import numpy as np


def compute_features(
    image_id: str,
    class_name: str,
    dapi_channel: np.ndarray,
    pi_channel: np.ndarray,
    nucleus_mask: np.ndarray,
    nuclei: list[dict],
    apoptosis_markers: dict,
) -> dict:
    """
    Extract 18-descriptor feature set for cell death classification.
    
    Features span:
    - 4 NEW apoptosis-specific: nuclear_fragmentation_index, cell_shrinkage_ratio, 
                                 membrane_blebbing_score, chromatin_condensation_proxy
    - 9 ADAPTED from BBBC021: cell_swelling_index, membrane_permeability_proxy, 
                               mean_intensity, total_intensity, intensity_variance,
                               area_covered_ratio, cell_count, density_cells_per_10k_px,
                               cell_area_mean
    - 5 SIZE DISTRIBUTION: cell_area_std, cell_area_median, small_cell_fraction,
                           medium_cell_fraction, large_cell_fraction
    """
    # Cell count and area statistics
    cell_count = len(nuclei)
    areas = np.array([n["area"] for n in nuclei], dtype=np.float32)
    
    if len(areas) > 0:
        cell_area_mean = float(np.mean(areas))
        cell_area_std = float(np.std(areas)) if len(areas) > 1 else 0.0
        cell_area_median = float(np.median(areas))
        
        # Cell area quartiles (for shrinkage/swelling detection)
        q25 = float(np.percentile(areas, 25))
        q75 = float(np.percentile(areas, 75))
        
        # Size distribution (thresholds: <50 = small, 50-200 = medium, >200 = large)
        small_cell_fraction = float(np.sum(areas < 50) / len(areas))
        medium_cell_fraction = float(np.sum((areas >= 50) & (areas < 200)) / len(areas))
        large_cell_fraction = float(np.sum(areas >= 200) / len(areas))
    else:
        cell_area_mean = 0.0
        cell_area_std = 0.0
        cell_area_median = 0.0
        q25 = 0.0
        q75 = 0.0
        small_cell_fraction = 0.0
        medium_cell_fraction = 0.0
        large_cell_fraction = 0.0
    
    # Image-level intensity statistics
    total_intensity = float(np.sum(dapi_channel))
    mean_intensity = float(np.mean(dapi_channel))
    intensity_variance = float(np.var(dapi_channel))
    
    # Coverage and density
    area_covered_px = int(np.count_nonzero(nucleus_mask))
    area_covered_ratio = float(area_covered_px / dapi_channel.size)
    density_cells_per_10k_px = float((cell_count / dapi_channel.size) * 10000)
    
    # Apoptosis-specific markers (from detect.py)
    nuclear_fragmentation_index = apoptosis_markers.get("nuclear_fragmentation_index", 0.0)
    membrane_blebbing_score = apoptosis_markers.get("membrane_blebbing_score", 0.0)
    chromatin_condensation_proxy = apoptosis_markers.get("chromatin_condensation_proxy", 0.0)
    
    # Necrosis markers
    membrane_permeability_proxy = apoptosis_markers.get("membrane_permeability_proxy", 0.0)
    
    # Cell swelling: deviation from baseline area (e.g., if q75 >> q25, cells are swollen)
    if q25 > 0:
        cell_swelling_index = float((q75 - q25) / q25)  # 0 = uniform, >1 = high variance
    else:
        cell_swelling_index = 0.0
    
    # Cell shrinkage: proportion of cells below median (apoptotic cells shrink)
    if cell_area_median > 0:
        cell_shrinkage_ratio = float(np.sum(areas < cell_area_median * 0.7) / len(areas)) if len(areas) > 0 else 0.0
    else:
        cell_shrinkage_ratio = 0.0
    
    return {
        "image_id": image_id,
        "class_name": class_name,
        # NEW apoptosis-specific (4)
        "nuclear_fragmentation_index": nuclear_fragmentation_index,
        "cell_shrinkage_ratio": cell_shrinkage_ratio,
        "membrane_blebbing_score": membrane_blebbing_score,
        "chromatin_condensation_proxy": chromatin_condensation_proxy,
        # ADAPTED necrosis markers (2)
        "cell_swelling_index": cell_swelling_index,
        "membrane_permeability_proxy": membrane_permeability_proxy,
        # ADAPTED intensity features (3)
        "mean_intensity": mean_intensity,
        "total_intensity": total_intensity,
        "intensity_variance": intensity_variance,
        # ADAPTED morphology (4)
        "area_covered_ratio": area_covered_ratio,
        "cell_count": float(cell_count),
        "density_cells_per_10k_px": density_cells_per_10k_px,
        "cell_area_mean": cell_area_mean,
        # SIZE DISTRIBUTION (5)
        "cell_area_std": cell_area_std,
        "cell_area_median": cell_area_median,
        "small_cell_fraction": small_cell_fraction,
        "medium_cell_fraction": medium_cell_fraction,
        "large_cell_fraction": large_cell_fraction,
    }
