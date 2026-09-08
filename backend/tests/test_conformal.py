"""
Unit tests for Split Conformal Prediction and temperature scaling.
"""

from app.models.conformal import ConformalPredictor


def test_conformal_singleton_for_dominant_class():
    predictor = ConformalPredictor(alpha=0.05)
    # High confidence prediction
    probs = {
        "No_DR": 0.96,
        "Mild_NPDR": 0.02,
        "Moderate_NPDR": 0.01,
        "Severe_NPDR": 0.005,
        "Proliferative_DR": 0.005,
    }
    c_set, size, is_singleton = predictor.compute_conformal_set(probs)
    assert is_singleton is True
    assert size == 1
    assert c_set == ["No_DR"]


def test_conformal_multi_class_for_ambiguous_prediction():
    predictor = ConformalPredictor(alpha=0.05)
    # Ambiguous between Moderate and Severe
    probs = {
        "Moderate_NPDR": 0.55,
        "Severe_NPDR": 0.40,
        "Proliferative_DR": 0.03,
        "Mild_NPDR": 0.01,
        "No_DR": 0.01,
    }
    c_set, size, is_singleton = predictor.compute_conformal_set(probs)
    assert is_singleton is False
    assert size == 2
    assert "Moderate_NPDR" in c_set
    assert "Severe_NPDR" in c_set


def test_temperature_scaling_calibrates_probabilities():
    predictor = ConformalPredictor(alpha=0.05, temperature=1.5)
    raw_probs = {
        "No_DR": 0.99,
        "Mild_NPDR": 0.01,
    }
    calibrated = predictor.calibrate_probabilities(raw_probs)
    # Softmax with temperature > 1 softens the peak
    assert calibrated["No_DR"] < 0.99
    assert calibrated["Mild_NPDR"] > 0.01
    assert abs(sum(calibrated.values()) - 1.0) < 1e-4
