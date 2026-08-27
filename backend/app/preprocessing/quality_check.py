"""
Image-quality gate — reject unusable fundus images before inference.

Checks:
  1. Laplacian-variance blur score  (< threshold → blurry)
  2. Mean brightness / exposure      (too dark or too bright → unusable)
  3. Contrast ratio                  (flat histogram → low contrast)

Returns a (passed: bool, issues: list[str]) tuple.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


# ── Thresholds (calibrate on your dataset in Phase 1 EDA) ────
BLUR_THRESHOLD: float = 50.0          # Laplacian variance below this → blurry
BRIGHTNESS_LOW: float = 30.0          # mean pixel value below this → underexposed
BRIGHTNESS_HIGH: float = 225.0        # mean pixel value above this → overexposed
CONTRAST_THRESHOLD: float = 30.0      # std-dev of pixel values below this → low contrast


def _laplacian_variance(gray: NDArray[np.uint8]) -> float:
    """Variance of Laplacian — higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_brightness(gray: NDArray[np.uint8]) -> float:
    return float(np.mean(gray))


def _contrast_stddev(gray: NDArray[np.uint8]) -> float:
    return float(np.std(gray))


def check_image_quality(
    image_bgr: NDArray[np.uint8],
    blur_thresh: float = BLUR_THRESHOLD,
    bright_low: float = BRIGHTNESS_LOW,
    bright_high: float = BRIGHTNESS_HIGH,
    contrast_thresh: float = CONTRAST_THRESHOLD,
) -> tuple[bool, list[str]]:
    """
    Run the quality gate on a BGR OpenCV image.

    Returns
    -------
    passed : bool
        True if image quality is acceptable for inference.
    issues : list[str]
        Human-readable list of quality problems (empty when passed=True).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    issues: list[str] = []

    # 1. Blur detection
    lap_var = _laplacian_variance(gray)
    if lap_var < blur_thresh:
        issues.append("blurry")

    # 2. Brightness / exposure
    brightness = _mean_brightness(gray)
    if brightness < bright_low:
        issues.append("underexposed")
    elif brightness > bright_high:
        issues.append("overexposed")

    # 3. Contrast
    contrast = _contrast_stddev(gray)
    if contrast < contrast_thresh:
        issues.append("low_contrast")

    passed = len(issues) == 0
    return passed, issues
