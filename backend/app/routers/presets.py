"""
Recruiter Demo Presets Router.
Provides pre-configured clinical test scenarios so recruiters, evaluators, and clinicians
can test the full diagnostic and multi-agent pipeline with a single click.
"""

from __future__ import annotations

import base64
import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/presets", tags=["Recruiter Demo Presets"])


class PresetCase(BaseModel):
    id: str
    title: str
    stage_label: str
    badge_color: str
    description: str
    clinical_findings: str
    thumbnail_b64: str = Field(..., description="Base64 PNG thumbnail")


def _generate_synthetic_fundus(case_type: str) -> str:
    """Generates an illustrative retinal fundus photograph base64 string."""
    size = 256
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # 1. Background dark space
    # 2. Retinal circle
    center = (size // 2, size // 2)
    radius = int(size * 0.44)
    cv2.circle(img, center, radius, (15, 30, 160), -1)  # Deep retinal red/orange in BGR

    # Soft radial gradient for fundus illumination
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    vignette = np.clip(1.0 - (dist_from_center / radius) ** 2 * 0.5, 0.2, 1.0)
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] * vignette, 0, 255).astype(np.uint8)

    # 3. Optic Disc (yellowish circle on the nasal side)
    disc_center = (int(size * 0.32), int(size * 0.50))
    cv2.circle(img, disc_center, int(size * 0.09), (70, 190, 230), -1)  # Warm yellow-orange

    # 4. Retinal vessels (arcing from optic disc)
    cv2.ellipse(img, disc_center, (int(size * 0.28), int(size * 0.24)), 30, 0, 160, (10, 15, 95), 2)
    cv2.ellipse(img, disc_center, (int(size * 0.30), int(size * 0.26)), -30, 0, 160, (10, 15, 95), 2)

    # 5. Macula / Fovea (darker red region temporal to disc)
    macula_center = (int(size * 0.62), int(size * 0.52))
    cv2.circle(img, macula_center, int(size * 0.05), (10, 18, 110), -1)

    # 6. Specific pathology additions based on case_type
    if case_type == "mild":
        # Isolated red dots (microaneurysms)
        cv2.circle(img, (int(size * 0.55), int(size * 0.42)), 2, (0, 0, 190), -1)
        cv2.circle(img, (int(size * 0.68), int(size * 0.60)), 2, (0, 0, 200), -1)
    elif case_type == "moderate":
        # Blot hemorrhages & hard exudates (bright yellow dots)
        cv2.circle(img, (int(size * 0.52), int(size * 0.38)), 4, (0, 0, 180), -1)
        cv2.circle(img, (int(size * 0.66), int(size * 0.44)), 5, (0, 0, 170), -1)
        cv2.circle(img, (int(size * 0.45), int(size * 0.68)), 4, (0, 0, 180), -1)
        # Exudates
        cv2.circle(img, (int(size * 0.58), int(size * 0.48)), 3, (80, 230, 255), -1)
        cv2.circle(img, (int(size * 0.62), int(size * 0.46)), 2, (80, 230, 255), -1)
    elif case_type == "pdr":
        # Extensive hemorrhages & neovascular fronds
        for pt in [(130, 110), (145, 140), (170, 95), (115, 160), (160, 165)]:
            cv2.circle(img, pt, 6, (0, 0, 160), -1)
        # Fragile tangled vessels near optic disc
        pts = np.array([[82, 128], [95, 115], [105, 120], [115, 105]], np.int32)
        cv2.polylines(img, [pts], False, (15, 20, 130), 2)
    elif case_type == "blur":
        # Heavy gaussian blur simulating cataract or camera shake
        img = cv2.GaussianBlur(img, (35, 35), 0)

    # Encode to PNG base64
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")


# Cache preset payloads
_PRESETS: list[PresetCase] = [
    PresetCase(
        id="preset_normal",
        title="Healthy Retina (Grade 0)",
        stage_label="No DR",
        badge_color="emerald",
        description="Normal fundus photograph with sharp optic disc borders, well-defined foveal avascular zone, and regular vascular calibers.",
        clinical_findings="Zero diabetic microvascular abnormalities. Annual routine screening protocol.",
        thumbnail_b64=_generate_synthetic_fundus("normal"),
    ),
    PresetCase(
        id="preset_mild",
        title="Early Stage (Grade 1 - Mild NPDR)",
        stage_label="Mild NPDR",
        badge_color="blue",
        description="Early non-proliferative changes with isolated capillary microaneurysms along superior temporal arcade.",
        clinical_findings="Meets ETDRS Mild NPDR criteria. Glycemic control review and 6-12 month follow-up.",
        thumbnail_b64=_generate_synthetic_fundus("mild"),
    ),
    PresetCase(
        id="preset_moderate",
        title="Established Disease (Grade 2 - Moderate NPDR)",
        stage_label="Moderate NPDR",
        badge_color="amber",
        description="Moderate retinopathy displaying multi-focal blot hemorrhages and clustered lipid exudates approaching macular boundary.",
        clinical_findings="Significant microvascular leakage. Ophthalmologic review within 2-3 months.",
        thumbnail_b64=_generate_synthetic_fundus("moderate"),
    ),
    PresetCase(
        id="preset_pdr",
        title="Sight-Threatening (Grade 4 - Proliferative DR)",
        stage_label="Proliferative DR",
        badge_color="red",
        description="Advanced neovascularization along major vascular arcades with high imminent risk of vitreous hemorrhage.",
        clinical_findings="Urgent 48-hour vitreoretinal specialist referral. Evaluation for anti-VEGF or panretinal photocoagulation.",
        thumbnail_b64=_generate_synthetic_fundus("pdr"),
    ),
    PresetCase(
        id="preset_reject",
        title="Quality Guardrail Reject (Cataract / Blur)",
        stage_label="Quality Reject",
        badge_color="purple",
        description="Dense media opacity causing severe image degradation below diagnostic sharpness threshold.",
        clinical_findings="Quality gate automatically triggers structured retake request. Prevents hallucinated grading.",
        thumbnail_b64=_generate_synthetic_fundus("blur"),
    ),
]


@router.get("", response_model=list[PresetCase])
def list_presets():
    """Returns curated recruiter demo test cases."""
    return _PRESETS
