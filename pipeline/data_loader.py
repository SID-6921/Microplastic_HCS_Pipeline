"""Data loading and simulation for A549 cell death phenotyping."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_bbbc014_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load BBBC014 metadata CSV (if available)."""
    return pd.read_csv(csv_path)


def simulate_bbbc014_dataset(
    num_images_per_class: int = 24,
    image_size: tuple[int, int] = (512, 512),
    random_seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], pd.DataFrame]:
    """
    Simulate A549 cell death dataset with BBBC014-like statistics.
    
    Classes:
    - 0: Viable (intact nuclei, low PI)
    - 1: Early apoptosis (fragmented nuclei, low-medium PI)
    - 2: Late apoptosis (highly fragmented, medium PI)
    - 3: Necrosis (condensed nuclei, high PI, swollen)
    
    Returns:
        dapi_images: List of DAPI channel images
        pi_images: List of PI channel images  
        metadata: DataFrame with image info and class labels
    """
    np.random.seed(random_seed)
    
    dapi_images = []
    pi_images = []
    metadata_list = []
    
    for class_id, class_name in enumerate(["viable", "early_apoptosis", "late_apoptosis", "necrosis"]):
        for idx in range(num_images_per_class):
            image_idx = class_id * num_images_per_class + idx
            image_id = f"img_{class_id}_{idx}"
            
            # Generate realistic synthetic images
            dapi = np.zeros(image_size, dtype=np.uint8)
            pi = np.zeros(image_size, dtype=np.uint8)
            
            # Class-specific parameters
            if class_id == 0:  # Viable
                num_nuclei = np.random.randint(8, 15)
                nucleus_size = np.random.randint(300, 500)
                pi_intensity = np.random.randint(10, 30)
                fragmentation = 0
            elif class_id == 1:  # Early apoptosis
                num_nuclei = np.random.randint(10, 18)
                nucleus_size = np.random.randint(200, 400)
                pi_intensity = np.random.randint(30, 80)
                fragmentation = 0.3
            elif class_id == 2:  # Late apoptosis
                num_nuclei = np.random.randint(15, 25)
                nucleus_size = np.random.randint(100, 250)
                pi_intensity = np.random.randint(80, 150)
                fragmentation = 0.6
            else:  # Necrosis (class_id == 3)
                num_nuclei = np.random.randint(6, 12)
                nucleus_size = np.random.randint(400, 700)
                pi_intensity = np.random.randint(150, 220)
                fragmentation = 0.1
            
            # Generate nuclei as Gaussian blobs
            for _ in range(num_nuclei):
                y = np.random.randint(50, image_size[0] - 50)
                x = np.random.randint(50, image_size[1] - 50)
                size = np.random.randint(int(nucleus_size * 0.8), int(nucleus_size * 1.2))
                
                # DAPI channel: Gaussian blob
                yy, xx = np.ogrid[:image_size[0], :image_size[1]]
                mask = (xx - x) ** 2 + (yy - y) ** 2 <= size
                dapi[mask] = np.maximum(dapi[mask], np.random.randint(100, 220))
                
                # PI channel: correlated with class
                if np.random.rand() < fragmentation:
                    # Fragmented nucleus: smaller, dimmer PI
                    pi[mask] = np.maximum(pi[mask], int(pi_intensity * 0.3))
                else:
                    pi[mask] = np.maximum(pi[mask], int(pi_intensity))
            
            # Add noise
            dapi = dapi.astype(np.int32)
            pi = pi.astype(np.int32)
            dapi += np.random.randint(0, 20, dapi.shape, dtype=np.int32)
            pi += np.random.randint(0, 20, pi.shape, dtype=np.int32)
            dapi = np.clip(dapi, 0, 255).astype(np.uint8)
            pi = np.clip(pi, 0, 255).astype(np.uint8)
            
            dapi_images.append(dapi)
            pi_images.append(pi)
            
            # plate_id: groups of 6 images per class → plate-level CV grouping
            plate_id = class_id * 100 + (idx // 6)

            metadata_list.append({
                "image_id": image_id,
                "class_id": class_id,
                "class_name": class_name,
                "image_index": image_idx,
                "plate_id": plate_id,
            })
    
    metadata = pd.DataFrame(metadata_list)
    return dapi_images, pi_images, metadata
