"""
Pydantic response schemas for the /predict endpoint.

Two response shapes:
  1. PredictionResponse  — good-quality image → full inference results
  2. QualityRejectResponse — poor-quality image → retake request (no inference)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Class labels (ICDR scale) ────────────────────────────────

CLASS_NAMES: list[str] = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]

NUM_CLASSES: int = len(CLASS_NAMES)


# ── Good-quality response ────────────────────────────────────

class PredictionResponse(BaseModel):
    """Returned when the uploaded image passes the quality gate."""

    prediction: str = Field(..., example="Moderate DR")
    class_index: int = Field(..., ge=0, le=4, example=2)
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.914)
    probabilities: Dict[str, float] = Field(
        ...,
        example={
            "No DR": 0.010,
            "Mild DR": 0.040,
            "Moderate DR": 0.914,
            "Severe DR": 0.025,
            "Proliferative DR": 0.011,
        },
    )
    certainty: str = Field(..., example="HIGH", description="HIGH or LOW")
    review_recommendation: str = Field(
        ...,
        example="Recommended",
        description="Recommended | Strongly Recommended",
    )
    image_quality: str = Field(default="good")
    gradcam_overlay: str = Field(
        ..., description="Base64-encoded PNG of the Grad-CAM heatmap overlay"
    )
    saliency_overlay: Optional[str] = Field(
        None, description="Base64-encoded PNG of the Gradient Saliency map overlay"
    )
    shap_overlay: Optional[str] = Field(
        None, description="Base64-encoded PNG of the SHAP Shapley attribution overlay"
    )
    model_version: str = Field(default="efficientnetv2s-v1")


# ── Poor-quality response ────────────────────────────────────

class QualityRejectResponse(BaseModel):
    """Returned when the uploaded image fails the quality gate."""

    image_quality: str = Field(default="poor")
    quality_issues: List[str] = Field(
        ..., example=["blurry", "underexposed"]
    )
    message: str = Field(
        default="Image quality insufficient for reliable grading. Please retake."
    )
