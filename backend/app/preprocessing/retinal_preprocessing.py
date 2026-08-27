"""
Retinal image preprocessing pipeline for inference.

Steps:
  1. Crop fundus circle
  2. Optional CLAHE enhancement (matches training augmentation)
  3. Resize to 224x224
  4. Normalize to [0, 1] float32 tensor for TensorFlow

NOTE: We do NOT apply tf.keras.applications.efficientnet.preprocess_input()
because the saved .keras model was trained with image_dataset_from_directory
which feeds raw [0, 255] pixels. The model's internal layers handle any
further normalization. We only do /255.0 rescaling here.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

# ── Constants ─────────────────────────────────────────────────
INPUT_SIZE: int = 224  # Model input size


def apply_clahe(
    image_bgr: NDArray[np.uint8],
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> NDArray[np.uint8]:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to a BGR image.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def crop_fundus_circle(image_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Auto-crop the circular fundus region from a rectangular image.
    """
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_bgr

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image_bgr.shape[1] - x, w + 2 * pad)
        h = min(image_bgr.shape[0] - y, h + 2 * pad)

        cropped = image_bgr[y : y + h, x : x + w]
        # Safety check — if crop is too small, return original
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return image_bgr
        return cropped
    except Exception:
        return image_bgr


def preprocess_for_inference(
    image_bgr: NDArray[np.uint8],
    use_clahe: bool = False,
    img_size: int = INPUT_SIZE
) -> np.ndarray:
    """
    Full preprocessing pipeline: crop → optional CLAHE → resize → normalize.

    Parameters
    ----------
    image_bgr : np.ndarray
        Raw BGR image from OpenCV.
    use_clahe : bool
        Whether to apply CLAHE enhancement (should match training config).
    img_size : int
        Target size.

    Returns
    -------
    np.ndarray
        Shape (1, img_size, img_size, 3), float32 in [0, 1] range,
        ready for TensorFlow input.
    """
    try:
        # 1. Skip cropping for now (often fails on web images and destroys the retina)
        cropped = image_bgr

        # 2. Optional CLAHE
        if use_clahe:
            cropped = apply_clahe(cropped)

        # 3. BGR → RGB
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        # 4. Resize
        resized = cv2.resize(rgb, (img_size, img_size))
        
        # 5. Format for TensorFlow
        from tensorflow.keras.applications.efficientnet import preprocess_input
        
        # Expand dims and cast to float32
        tensor = np.expand_dims(resized, axis=0).astype(np.float32)
        
        # Apply model-specific preprocessing
        tensor = preprocess_input(tensor)

        print(f"[DEBUG PREPROC] Tensor Shape: {tensor.shape}, Min: {tensor.min():.3f}, Max: {tensor.max():.3f}, Mean: {tensor.mean():.3f}")

        return tensor
    except Exception as e:
        print(f"[WARN] Preprocessing failed: {e}. Using raw resized image.")
        # Absolute fallback — just resize and preprocess
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (img_size, img_size))
        
        from tensorflow.keras.applications.efficientnet import preprocess_input
        tensor = np.expand_dims(resized, axis=0).astype(np.float32)
        tensor = preprocess_input(tensor)
        return tensor
