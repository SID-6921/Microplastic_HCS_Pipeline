from __future__ import annotations

import numpy as np

from pipeline.data_loader import simulate_bbbc014_dataset
from pipeline.detect import detect_nuclei, measure_cell_morphology
from pipeline.features import compute_features
from pipeline.preprocess import clean_image, normalize_image, resize_image


def test_compute_features_returns_expected_schema() -> None:
    dapi = np.zeros((32, 32), dtype=np.uint8)
    pi = np.zeros((32, 32), dtype=np.uint8)
    nucleus_mask = np.zeros((32, 32), dtype=np.uint8)
    nucleus_mask[8:12, 8:12] = 255
    nuclei = [
        {
            "area": 16,
            "centroid": (9.5, 9.5),
            "pi_intensity": 42.0,
            "is_fragmented": False,
            "fragment_count": 1,
        }
    ]
    apoptosis_markers = {
        "nuclear_fragmentation_index": 0.25,
        "membrane_blebbing_score": 0.10,
        "membrane_permeability_proxy": 42.0,
        "chromatin_condensation_proxy": 1.5,
    }

    features = compute_features(
        image_id="img_0",
        class_id=0,
        class_name="viable",
        dapi_channel=dapi,
        pi_channel=pi,
        nucleus_mask=nucleus_mask,
        nuclei=nuclei,
        apoptosis_markers=apoptosis_markers,
    )

    expected_keys = {
        "image_id",
        "class_id",
        "class_name",
        "nuclear_fragmentation_index",
        "cell_shrinkage_ratio",
        "membrane_blebbing_score",
        "chromatin_condensation_proxy",
        "cell_swelling_index",
        "membrane_permeability_proxy",
        "mean_intensity",
        "total_intensity",
        "intensity_variance",
        "area_covered_ratio",
        "cell_count",
        "density_cells_per_10k_px",
        "cell_area_mean",
        "cell_area_std",
        "cell_area_median",
        "small_cell_fraction",
        "medium_cell_fraction",
        "large_cell_fraction",
    }

    assert expected_keys.issubset(features)
    assert features["image_id"] == "img_0"
    assert features["class_id"] == 0
    assert features["cell_count"] == 1.0
    assert isinstance(features["mean_intensity"], float)
    assert isinstance(features["total_intensity"], float)


def test_preprocess_and_detection_smoke() -> None:
    dapi_images, pi_images, metadata = simulate_bbbc014_dataset(num_images_per_class=1, random_seed=7)
    dapi = clean_image(normalize_image(resize_image(dapi_images[0], (64, 64))))
    pi = clean_image(normalize_image(resize_image(pi_images[0], (64, 64))))

    mask, nuclei = detect_nuclei(dapi, pi)
    markers = measure_cell_morphology(dapi, pi, nuclei)

    assert mask.shape == dapi.shape
    assert isinstance(nuclei, list)
    assert set(markers) == {
        "nuclear_fragmentation_index",
        "membrane_blebbing_score",
        "membrane_permeability_proxy",
        "chromatin_condensation_proxy",
    }
    assert len(metadata) == 4
