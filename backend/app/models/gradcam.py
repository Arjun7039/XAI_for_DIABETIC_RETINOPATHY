"""
Grad-CAM heatmap generator for RetinaScreen AI (TensorFlow / Keras 3).
Optimized with graph caching for 512MB RAM server deployment.
"""

from __future__ import annotations

import base64
import gc
import cv2
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

_GRAD_MODEL_CACHE: dict[int, tf.keras.Model] = {}


def _find_last_conv_layer(model: tf.keras.Model) -> str | None:
    """Walk the model graph to find the last Conv2D layer name."""
    last_conv_name = None
    target_model = model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            target_model = layer
            break

    for layer in target_model.layers:
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            last_conv_name = layer.name
        else:
            try:
                out_shape = layer.output.shape
                if len(out_shape) == 4:
                    last_conv_name = layer.name
            except (AttributeError, RuntimeError):
                continue

    return last_conv_name


def _get_or_create_grad_model(model: tf.keras.Model) -> tf.keras.Model | None:
    """Cache the sub-graph model to avoid creating new Keras models on every request."""
    model_id = id(model)
    if model_id in _GRAD_MODEL_CACHE:
        return _GRAD_MODEL_CACHE[model_id]

    last_conv_layer_name = _find_last_conv_layer(model)
    if not last_conv_layer_name:
        return None

    # First try building directly on top-level model
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[last_conv_layer.output, model.output],
        )
        _GRAD_MODEL_CACHE[model_id] = grad_model
        return grad_model
    except Exception:
        pass

    # Fallback to sub-model graph
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                last_conv_layer = layer.get_layer(last_conv_layer_name)
                grad_model = tf.keras.models.Model(
                    inputs=layer.input,
                    outputs=[last_conv_layer.output, layer.output],
                )
                _GRAD_MODEL_CACHE[model_id] = grad_model
                return grad_model
            except Exception:
                continue

    return None


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
        grad_model = _get_or_create_grad_model(model)
        if grad_model is None:
            raise ValueError("Could not construct Grad-CAM sub-model graph.")

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tensor)
            if target_category is None:
                cat_idx = int(tf.argmax(predictions[0]).numpy())
            else:
                cat_idx = int(target_category)
            loss = predictions[:, cat_idx]

        grads = tape.gradient(loss, conv_outputs)
        del tape  # Release tape memory immediately

        if grads is None:
            raise ValueError("Gradient computation returned None.")

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalize
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.numpy()
        heatmap_resized = cv2.resize(heatmap_np, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

        _, buffer = cv2.imencode(".png", overlay)
        encoded = base64.b64encode(buffer).decode("utf-8")

        gc.collect()
        return encoded

    except Exception as e:
        print(f"[WARN] Grad-CAM generation failed: {e}")
        return _generate_dummy_overlay(original_bgr, "Grad-CAM")


def _generate_dummy_overlay(original_bgr: NDArray[np.uint8], text: str = "Grad-CAM") -> str:
    """Generates a fallback visual overlay."""
    h, w = original_bgr.shape[:2]
    overlay = original_bgr.copy()
    cv2.putText(
        overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    _, buffer = cv2.imencode(".png", overlay)
    return base64.b64encode(buffer).decode("utf-8")
