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

from app.agents.schemas import MultiAgentConsensusReport


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
    conformal_prediction_set: List[str] = Field(
        default_factory=list,
        example=["Moderate DR"],
        description="95% statistical coverage prediction set derived via Split Conformal Prediction",
    )
    conformal_coverage: float = Field(
        default=0.95,
        description="Coverage guarantee probability (1 - alpha)",
    )
    image_quality: str = Field(default="good")
    gradcam_overlay: str = Field(
        ..., description="Base64-encoded PNG of the Grad-CAM++ heatmap overlay"
    )
    saliency_overlay: Optional[str] = Field(
        None, description="Base64-encoded PNG of the Gradient Saliency map overlay"
    )
    shap_overlay: Optional[str] = Field(
        None, description="Base64-encoded PNG of the SHAP Shapley attribution overlay"
    )
    model_version: str = Field(default="efficientnetv2s-v1")
    agentic_findings: Optional[MultiAgentConsensusReport] = Field(
        None, description="Synthesized multi-agent clinical consultation findings"
    )
    execution_telemetry_ms: Optional[Dict[str, float]] = Field(
        None, description="Component latency breakdown in milliseconds"
    )


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


# ── Hospital Patient Queue Triage Schemas ─────────────────────

class PatientTriageCard(BaseModel):
    """Represents a triaged patient in a hospital queue."""

    patient_id: str = Field(..., example="PT-104")
    patient_name: str = Field(default="Anonymous Patient", example="John Doe (OD)")
    prediction: str = Field(..., example="Proliferative DR")
    class_index: int = Field(..., ge=0, le=4, example=4)
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.965)
    triage_priority: str = Field(..., example="P1_CRITICAL", description="P1_CRITICAL | P2_URGENT | P3_ROUTINE")
    priority_label: str = Field(..., example="P1 - Immediate Referral (<24h)")
    priority_color: str = Field(..., example="red", description="red | amber | green")
    referral_urgency: str = Field(..., example="Immediate referral to retina specialist within 24-48 hours")
    recommended_action: str = Field(..., example="Panretinal photocoagulation (PRP) evaluation")
    conformal_set: List[str] = Field(default_factory=list, example=["Proliferative DR"])
    icd10_code: str = Field(default="E11.359")
    latency_ms: float = Field(default=0.0, example=85.4)
    thumbnail_b64: Optional[str] = Field(None, description="Base64 thumbnail for queue card preview")


class BatchTriageResponse(BaseModel):
    """Response returned when processing a multi-patient triage queue."""

    total_patients: int = Field(..., example=5)
    critical_p1_count: int = Field(..., example=1)
    urgent_p2_count: int = Field(..., example=2)
    routine_p3_count: int = Field(..., example=2)
    total_latency_ms: float = Field(..., example=412.5)
    avg_latency_ms: float = Field(..., example=82.5)
    patients: List[PatientTriageCard] = Field(
        ..., description="Patients sorted strictly in order of clinical severity / triage priority"
    )

