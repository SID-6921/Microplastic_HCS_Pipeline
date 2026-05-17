"""Generate a harder synthetic dataset (more images, extra noise/blur/overlap),
extract features, run CV and generate figures.
"""
from pathlib import Path
import numpy as np
import random
import cv2
import pandas as pd

from data_loader import simulate_bbbc014_dataset
from full_dataset_advanced_pipeline import preprocess_and_extract_features

OUT = Path('results')
OUT.mkdir(exist_ok=True)

def make_harder(dapi_images, pi_images, severity=1.5, blur_prob=0.3, mix_prob=0.2):
    new_dapi = []
    new_pi = []
    N = len(dapi_images)
    for i in range(N):
        im = dapi_images[i].astype(np.float32)
        pi = pi_images[i].astype(np.float32)
        # Add Gaussian noise (severity scaled)
        im += np.random.normal(0, 10 * severity, im.shape)
        pi += np.random.normal(0, 10 * severity, pi.shape)
        # Random blur
        if random.random() < blur_prob:
            k = random.choice([3,5,7])
            im = cv2.GaussianBlur(im, (k,k), 0)
            pi = cv2.GaussianBlur(pi, (k,k), 0)
        # Slight class overlap: mix with a random other image's PI
        if random.random() < mix_prob:
            j = random.randrange(N)
            pi = 0.7 * pi + 0.3 * pi_images[j]
        # Clip
        im = np.clip(im, 0, 255).astype(np.uint8)
        pi = np.clip(pi, 0, 255).astype(np.uint8)
        new_dapi.append(im)
        new_pi.append(pi)
    return new_dapi, new_pi


def main():
    # Generate more images per class for a harder test
    num_images_per_class = 48
    dapi_images, pi_images, metadata = simulate_bbbc014_dataset(num_images_per_class=num_images_per_class, random_seed=123)

    dapi_images_h, pi_images_h = make_harder(dapi_images, pi_images, severity=2.0, blur_prob=0.4, mix_prob=0.35)

    # Extract features using existing pipeline function
    features_df = preprocess_and_extract_features(dapi_images_h, pi_images_h, metadata)
    features_df.to_csv(OUT / 'features.csv', index=False)
    print('Harder features saved to results/features.csv')

    # Run existing CV & figure generation scripts
    import subprocess
    subprocess.run(['python', 'cv_evaluation.py'], check=True)
    subprocess.run(['python', 'rf_regularization_cv.py'], check=True)
    subprocess.run(['python', 'generate_figures.py'], check=True)
    print('Harder pipeline complete')

if __name__ == '__main__':
    main()
