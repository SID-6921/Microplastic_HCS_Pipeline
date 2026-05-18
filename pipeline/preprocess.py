"""Preprocessing module: resize, normalize, clean images."""
from __future__ import annotations

import cv2
import numpy as np


def resize_image(image: np.ndarray, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Resize image to target size using area interpolation."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 range."""
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def clean_image(image: np.ndarray) -> np.ndarray:
    """Apply light denoising and local contrast enhancement (CLAHE)."""
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)
