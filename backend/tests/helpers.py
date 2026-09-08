"""
Shared test helper utilities for synthetic fundus image generation.
"""

import cv2
import numpy as np


def create_synthetic_fundus(case_type: str = "normal") -> np.ndarray:
    """Creates a synthetic OpenCV BGR fundus image for testing."""
    size = 224
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Retinal circle
    center = (size // 2, size // 2)
    radius = int(size * 0.44)
    cv2.circle(img, center, radius, (20, 35, 175), -1)  # Orange-red retinal color

    # Optic disc
    cv2.circle(img, (int(size * 0.35), int(size * 0.5)), int(size * 0.08), (60, 180, 230), -1)

    # Macula
    cv2.circle(img, (int(size * 0.65), int(size * 0.52)), int(size * 0.06), (10, 20, 120), -1)

    if case_type == "blur":
        img = cv2.GaussianBlur(img, (45, 45), 0)
    elif case_type == "document":
        # Plain white/grey document with dark text
        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        cv2.putText(img, "PATIENT REPORT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    elif case_type == "lesions":
        # Add microaneurysms and blot hemorrhages
        cv2.circle(img, (110, 90), 3, (0, 0, 180), -1)
        cv2.circle(img, (130, 140), 4, (0, 0, 190), -1)

    return img


def to_jpeg_bytes(img_bgr: np.ndarray) -> bytes:
    """Encodes OpenCV image to JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", img_bgr)
    return buffer.tobytes()
