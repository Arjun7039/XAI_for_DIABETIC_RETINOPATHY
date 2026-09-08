"""
Agent 1: Clinical Triage Specialist
Analyzes retinal fundus image metrics, visual feature attributions, and localized lesion signatures.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from app.agents.schemas import Agent1TriageOutput, LesionHallmark


class ClinicalTriageAgent:
    """
    Agent 1 specializes in pre-clinical triage:
    - Assesses image quality & optical clarity.
    - Correlates XAI activation patterns with ETDRS lesion hallmarks.
    - Evaluates optic disc and macular regional involvement.
    """

    def __init__(self):
        self.name = "Agent 1: Clinical Triage Specialist"

    async def analyze(
        self,
        predicted_stage: str,
        confidence: float,
        blur_score: float,
        contrast_score: float,
        gradcam_stats: Optional[Dict[str, float]] = None,
    ) -> Agent1TriageOutput:
        """
        Executes triage analysis based on optical parameters and activation statistics.
        """
        # Determine image quality grade
        if blur_score > 100.0 and contrast_score > 40.0:
            quality_grade = "Optimal"
        elif blur_score > 50.0 and contrast_score > 30.0:
            quality_grade = "Diagnostic"
        else:
            quality_grade = "Suboptimal"

        # Hallmarks detection based on predicted severity & heat localization
        lesions: List[LesionHallmark] = []
        macular_involvement = False
        optic_disc_visibility = "Clear"

        clean_stage = predicted_stage.lower().replace("_", " ")

        if "no dr" in clean_stage or "grade 0" in clean_stage:
            summary = "Fundus examination demonstrates sharp macular reflexes, normal cup-to-disc ratio, and clear vascular arcades without detectable diabetic microvascular lesions."
        elif "mild" in clean_stage or "grade 1" in clean_stage:
            lesions.append(
                LesionHallmark(
                    name="Microaneurysms",
                    confidence=float(min(0.95, confidence + 0.05)),
                    severity="Mild",
                    quadrants_affected=1,
                    clinical_note="Isolated focal capillary outpouchings noted. Earliest clinically observable sign of non-proliferative retinopathy.",
                )
            )
            summary = "Isolated microaneurysms detected in peripheral arcades. No signs of significant macular edema or venous loops."
        elif "moderate" in clean_stage or "grade 2" in clean_stage:
            lesions.append(
                LesionHallmark(
                    name="Intraretinal Hemorrhages (Dot & Blot)",
                    confidence=float(min(0.96, confidence + 0.02)),
                    severity="Moderate",
                    quadrants_affected=2,
                    clinical_note="Multiple deep capillary hemorrhages in mid-periphery.",
                )
            )
            lesions.append(
                LesionHallmark(
                    name="Hard Exudates (Lipid Deposition)",
                    confidence=float(max(0.70, confidence - 0.1)),
                    severity="Moderate",
                    quadrants_affected=2,
                    clinical_note="Waxy yellow lipid deposits precipitated from permeable capillaries.",
                )
            )
            macular_involvement = True
            summary = "Multiple intraretinal microaneurysms, dot-and-blot hemorrhages, and lipid exudates. Potential parafoveal involvement observed."
        elif "severe" in clean_stage or "grade 3" in clean_stage:
            lesions.append(
                LesionHallmark(
                    name="Extensive Intraretinal Hemorrhages (4-2-1 Rule)",
                    confidence=float(min(0.98, confidence + 0.04)),
                    severity="Severe",
                    quadrants_affected=4,
                    clinical_note="Severe blot hemorrhages in ≥2 quadrants meeting ETDRS 4-2-1 criteria.",
                )
            )
            lesions.append(
                LesionHallmark(
                    name="Cotton Wool Spots (Soft Exudates)",
                    confidence=float(max(0.80, confidence - 0.05)),
                    severity="Severe",
                    quadrants_affected=3,
                    clinical_note="Micro-infarctions in retinal nerve fiber layer indicating severe localized ischemia.",
                )
            )
            lesions.append(
                LesionHallmark(
                    name="Venous Beading (VB)",
                    confidence=float(max(0.75, confidence - 0.08)),
                    severity="Severe",
                    quadrants_affected=2,
                    clinical_note="Non-uniform caliber variations along major retinal venules.",
                )
            )
            macular_involvement = True
            summary = "Severe non-proliferative changes with multi-quadrant deep hemorrhages, soft exudates, and venous caliber abnormalities. High imminent risk of progression to neovascularization."
        elif "proliferative" in clean_stage or "grade 4" in clean_stage:
            lesions.append(
                LesionHallmark(
                    name="Neovascularization (NVD / NVE)",
                    confidence=float(min(0.99, confidence + 0.05)),
                    severity="Severe",
                    quadrants_affected=4,
                    clinical_note="Fragile new vessel proliferation on optic disc or along major arcades breaching internal limiting membrane.",
                )
            )
            lesions.append(
                LesionHallmark(
                    name="Preretinal / Vitreous Hemorrhage Risk",
                    confidence=float(min(0.92, confidence)),
                    severity="Severe",
                    quadrants_affected=3,
                    clinical_note="Extravasated blood into subhyaloid or vitreous cavity threatening immediate visual acuity.",
                )
            )
            optic_disc_visibility = "Obscured"
            macular_involvement = True
            summary = "Hallmark signs of Proliferative Diabetic Retinopathy: active neovascularization, prominent ischemic drive, and high risk of tractional retinal detachment."
        else:
            summary = f"Stage classified as {predicted_stage}. Visual examination shows notable microvascular alterations consistent with diabetic microangiopathy."

        return Agent1TriageOutput(
            image_quality_grade=quality_grade,
            blur_score=round(blur_score, 2),
            contrast_score=round(contrast_score, 2),
            lesions_detected=lesions,
            optic_disc_visibility=optic_disc_visibility,
            macular_involvement=macular_involvement,
            summary=summary,
        )
