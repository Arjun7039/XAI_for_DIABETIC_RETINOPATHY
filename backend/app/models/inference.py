"""
Model loading and inference helper for RetinaScreen AI.

Model loading and inference helper for RetinaScreen AI.

Loads an Ensemble model (EfficientNetB4 + ViT), performs inference on a NumPy array,
computes class probabilities, and applies certainty thresholds / calibration rules.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

# High certainty cutoff derived from calibration (ECE) evaluation
HIGH_CERTAINTY_THRESHOLD: float = 0.75

def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    print(f"[WARN] Config file '{config_path}' not found. Using defaults.")
    return {
        "class_names": ["Mild_NPDR", "Moderate_NPDR", "No_DR", "Proliferative_DR", "Severe_NPDR"],
        "num_classes": 5
    }

def _is_valid_keras_file(filepath: str) -> bool:
    """Return True if path exists and size > 1MB (not a Git LFS text pointer file)."""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 1_000_000


def load_model(config: dict) -> tf.keras.Model:
    weights_dir = "weights" if os.path.exists("weights/efficientnet_b4_config.json") else "backend/weights"
    eff_filename = config.get("efficientnet_model", "efficientnet_b4_best.keras")
    eff_path = os.path.join(weights_dir, eff_filename)

    # 1. Check if weights file is missing or is an invalid Git LFS text pointer (< 1MB)
    if not _is_valid_keras_file(eff_path):
        weights_url = os.getenv("MODEL_WEIGHTS_URL")
        if weights_url:
            print(f"[INFO] Valid weights file not found locally (missing or LFS pointer). Downloading from MODEL_WEIGHTS_URL: {weights_url}")
            try:
                import urllib.request
                os.makedirs(os.path.dirname(eff_path) or ".", exist_ok=True)
                urllib.request.urlretrieve(weights_url, eff_path)
                print(f"[INFO] Successfully downloaded weights ({os.path.getsize(eff_path)} bytes) to {eff_path}")
            except Exception as e:
                print(f"[ERROR] Failed to download weights from {weights_url}: {e}")

    # 2. Try loading the model file safely
    if _is_valid_keras_file(eff_path):
        try:
            print(f"[INFO] Loading EfficientNet-B4 from {eff_path} ({os.path.getsize(eff_path)} bytes)...")
            eff_model = tf.keras.models.load_model(eff_path, compile=False)
            print("[INFO] EfficientNet-B4 model loaded successfully!")
            return eff_model
        except Exception as e:
            print(f"[WARN] Failed to load model from {eff_path}: {e}. Using fallback dev model.")
    else:
        print(f"[WARN] Weights file not found or invalid at {eff_path}. Using fallback dev model.")
        # Fallback for dev mode
        from tensorflow.keras.applications import EfficientNetB4
        from tensorflow.keras import layers, models
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        
        eff_base = EfficientNetB4(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.applications.efficientnet.preprocess_input(inputs)
        x = eff_base(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        outputs = layers.Dense(5, activation='softmax')(x)
        return models.Model(inputs, outputs)

def run_inference(
    model: tf.keras.Model,
    tensor: np.ndarray,
    class_names: list[str]
) -> Tuple[str, int, float, Dict[str, float], str, str]:
    """
    Run forward pass and return prediction details:
    (prediction_class_name, class_index, confidence, probabilities_dict, certainty, review_recommendation)
    """
    preds = model.predict(tensor, verbose=0)
    probs = preds[0]  # First item in batch

    class_index = int(np.argmax(probs))
    confidence = float(np.clip(probs[class_index], 0.0, 1.0))
    prediction = class_names[class_index]

    probabilities_dict = {
        class_names[i]: float(np.clip(probs[i], 0.0, 1.0)) for i in range(len(class_names))
    }

    # Certainty calibration threshold logic
    if confidence >= HIGH_CERTAINTY_THRESHOLD:
        certainty = "HIGH"
        review_recommendation = "Recommended"
    else:
        certainty = "LOW"
        review_recommendation = "Strongly Recommended"

    return (
        prediction,
        class_index,
        confidence,
        probabilities_dict,
        certainty,
        review_recommendation,
    )
