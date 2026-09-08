"""
Unit tests for the optical quality gate and non-retinal image rejection.
"""

from app.preprocessing.quality_check import check_image_quality, get_image_quality_metrics
from helpers import create_synthetic_fundus


def test_quality_gate_passes_sharp_retina():
    fundus = create_synthetic_fundus("normal")
    passed, issues = check_image_quality(fundus)
    assert passed is True
    assert len(issues) == 0


def test_quality_gate_rejects_blurry_image():
    blurry = create_synthetic_fundus("blur")
    passed, issues = check_image_quality(blurry)
    assert passed is False
    assert "blurry" in issues


def test_quality_gate_rejects_document_image():
    doc = create_synthetic_fundus("document")
    passed, issues = check_image_quality(doc)
    assert passed is False
    assert "non_retinal_image" in issues


def test_quality_metrics_telemetry():
    fundus = create_synthetic_fundus("normal")
    metrics = get_image_quality_metrics(fundus)
    assert "blur_score" in metrics
    assert "contrast_score" in metrics
    assert "red_ratio" in metrics
    assert metrics["passed"] is True
    assert metrics["blur_score"] > 0
