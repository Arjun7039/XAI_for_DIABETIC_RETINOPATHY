"""
Grad-CAM heatmap generator for RetinaScreen AI (TensorFlow / Keras 3).
"""

from __future__ import annotations

import base64
import cv2
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


def _find_last_conv_layer(model: tf.keras.Model) -> str | None:
    """
    Walk the model graph to find the last Conv2D layer name.
    Compatible with Keras 3 (no layer.output_shape attribute).
    """
    last_conv_name = None

    # If the model wraps a nested backbone (e.g., EfficientNet functional model),
    # check inner layers first.
    target_model = model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            target_model = layer
            break

    for layer in target_model.layers:
        # Check by layer type — reliable across Keras 2 & 3
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            last_conv_name = layer.name
        # Also check for layers whose output is 4D (batch, h, w, c) via shape
        else:
            try:
                out_shape = layer.output.shape
                if len(out_shape) == 4:
                    last_conv_name = layer.name
            except (AttributeError, RuntimeError):
                continue

    return last_conv_name


def generate_gradcam_overlay(
    model: tf.keras.Model,
    tensor: np.ndarray,
    original_bgr: NDArray[np.uint8],
    target_category: int | None = None,
) -> str:
    """
    Generates a Grad-CAM heatmap overlay for the specified model and target class category.
    Returns Base64-encoded PNG image string.
    """
    h, w = original_bgr.shape[:2]

    try:
        last_conv_layer_name = _find_last_conv_layer(model)

        if not last_conv_layer_name:
            raise ValueError("Could not find a convolutional layer for Grad-CAM.")

        # Find the model that actually contains the conv layer
        target_model = model
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    layer.get_layer(last_conv_layer_name)
                    target_model = layer
                    break
                except ValueError:
                    continue

        last_conv_layer = target_model.get_layer(last_conv_layer_name)

        # Build a mini-model using the target_model (bypasses disconnected parent graphs)
        grad_model = tf.keras.models.Model(
            inputs=target_model.input,
            outputs=[last_conv_layer.output, target_model.output],
        )

        with tf.GradientTape() as tape:
            # If target_model is an inner model, we should pass the tensor through
            # the parent model's preprocessing layers first. But for Keras 3 standard
            # wrappers, passing it directly is usually mathematically identical if 
            # there are no prep layers.
            conv_outputs, predictions = grad_model(tensor)
            if target_category is None:
                target_category = tf.argmax(predictions[0])
            loss = predictions[:, target_category]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            raise ValueError("Gradient computation returned None — model graph may be disconnected.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

        _, buffer = cv2.imencode(".png", overlay)
        encoded = base64.b64encode(buffer).decode("utf-8")
        return encoded

    except Exception as e:
        print(f"[WARN] Grad-CAM generation failed: {e}")
        return _generate_dummy_overlay(original_bgr, "Grad-CAM Unavailable")


def _generate_dummy_overlay(original_bgr: NDArray[np.uint8], text: str = "Grad-CAM") -> str:
    """Generates a fallback visual overlay."""
    h, w = original_bgr.shape[:2]
    overlay = original_bgr.copy()
    cv2.putText(
        overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer).decode("utf-8")
