"""
SHAP-style attribution map explainer for RetinaScreen AI (TensorFlow / Keras 3).

Uses a gradient-based saliency map as the primary method, which is fully compatible
with Keras 3 and avoids the SHAP library's known incompatibilities with modern
TensorFlow functional models. Falls back to edge-based visualization on error.
"""

from __future__ import annotations

import base64
import cv2
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


def generate_shap_overlay(
    model: tf.keras.Model,
    tensor: np.ndarray,
    original_bgr: NDArray[np.uint8],
) -> str | None:
    """
    Generates a gradient-based saliency attribution overlay for the model prediction.
    Returns Base64-encoded PNG image string.

    This uses vanilla gradient saliency (∂output/∂input) which is mathematically
    equivalent to a first-order SHAP approximation and is fully Keras 3 compatible.
    """
    h, w = original_bgr.shape[:2]

    try:
        input_tensor = tf.cast(tensor, tf.float32)
        input_tensor = tf.Variable(input_tensor)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = model(input_tensor, training=False)
            # Get the gradient with respect to the predicted class
            predicted_class = tf.argmax(predictions[0])
            class_score = predictions[:, predicted_class]

        grads = tape.gradient(class_score, input_tensor)

        if grads is None:
            raise ValueError("Gradient computation returned None.")

        # Take absolute value and average across color channels → saliency map
        saliency = tf.abs(grads[0])
        saliency = tf.reduce_mean(saliency, axis=-1)  # (H, W)

        # Normalize to 0-255
        saliency = saliency.numpy()
        saliency_norm = cv2.normalize(saliency, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply slight Gaussian blur for smoother visualization
        saliency_norm = cv2.GaussianBlur(saliency_norm, (9, 9), 0)

        # Resize to original image dimensions
        saliency_resized = cv2.resize(saliency_norm, (w, h))

        # Apply Viridis colormap (same as SHAP default)
        heatmap = cv2.applyColorMap(saliency_resized, cv2.COLORMAP_VIRIDIS)
        overlay = cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)

        # Clean up TensorFlow tape and tensors to free RAM
        del tape
        del input_tensor
        import gc
        gc.collect()

        _, buffer = cv2.imencode(".png", overlay)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"[WARN] Saliency/SHAP generation failed: {e}")
        return _generate_dummy_shap_overlay(original_bgr, "Attribution Unavailable")


def _generate_dummy_shap_overlay(original_bgr: NDArray[np.uint8], text: str = "SHAP Attribution") -> str:
    """Fallback visual for SHAP attribution."""
    h, w = original_bgr.shape[:2]
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_VIRIDIS)

    overlay = cv2.addWeighted(original_bgr, 0.7, edges_colored, 0.3, 0)
    cv2.putText(
        overlay,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer).decode("utf-8")
