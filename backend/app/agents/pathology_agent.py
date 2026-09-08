"""
Agent 2: Diagnostic & Staging Specialist
Synthesizes model probabilities, ICDR classification criteria, ICD-10 medical coding, and ETDRS progression risk.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.agents.schemas import Agent2PathologyOutput, DiagnosticDifferential

ICD10_MAP: Dict[str, str] = {
    "No_DR": "E11.9 (Type 2 Diabetes Mellitus without complications / Retinal screen normal)",
    "No DR": "E11.9 (Type 2 Diabetes Mellitus without complications / Retinal screen normal)",
    "Mild_NPDR": "E11.329 (Type 2 Diabetes Mellitus with mild nonproliferative diabetic retinopathy)",
    "Mild DR": "E11.329 (Type 2 Diabetes Mellitus with mild nonproliferative diabetic retinopathy)",
    "Moderate_NPDR": "E11.339 (Type 2 Diabetes Mellitus with moderate nonproliferative diabetic retinopathy)",
    "Moderate DR": "E11.339 (Type 2 Diabetes Mellitus with moderate nonproliferative diabetic retinopathy)",
    "Severe_NPDR": "E11.349 (Type 2 Diabetes Mellitus with severe nonproliferative diabetic retinopathy)",
    "Severe DR": "E11.349 (Type 2 Diabetes Mellitus with severe nonproliferative diabetic retinopathy)",
    "Proliferative_DR": "E11.359 (Type 2 Diabetes Mellitus with proliferative diabetic retinopathy)",
    "Proliferative DR": "E11.359 (Type 2 Diabetes Mellitus with proliferative diabetic retinopathy)",
}

ETDRS_RISK_MAP: Dict[str, str] = {
    "No_DR": "Low (<1% estimated 1-year progression to sight-threatening DR)",
    "No DR": "Low (<1% estimated 1-year progression to sight-threatening DR)",
    "Mild_NPDR": "Low (~5% estimated 1-year progression to proliferative disease)",
    "Mild DR": "Low (~5% estimated 1-year progression to proliferative disease)",
    "Moderate_NPDR": "Moderate (12–27% estimated 1-year progression to severe/proliferative stage)",
    "Moderate DR": "Moderate (12–27% estimated 1-year progression to severe/proliferative stage)",
    "Severe_NPDR": "High (~52% estimated 1-year progression to proliferative retinopathy)",
    "Severe DR": "High (~52% estimated 1-year progression to proliferative retinopathy)",
    "Proliferative_DR": "Critical (>75% risk of profound visual loss without panretinal photocoagulation or anti-VEGF therapy)",
    "Proliferative DR": "Critical (>75% risk of profound visual loss without panretinal photocoagulation or anti-VEGF therapy)",
}


class DiagnosticPathologyAgent:
    """
    Agent 2 synthesizes quantitative classification data with medical ontology and international diagnostic criteria.
    """

    def __init__(self):
        self.name = "Agent 2: Diagnostic & Staging Specialist"

    async def stage(
        self,
        predicted_stage: str,
        probabilities: Dict[str, float],
        conformal_set: List[str],
    ) -> Agent2PathologyOutput:
        """
        Derives clinical ICD-10 coding, differential diagnosis, and ETDRS progression risk.
        """
        # Look up primary ICD-10 code
        icd10_code = ICD10_MAP.get(
            predicted_stage,
            "E11.319 (Type 2 Diabetes Mellitus with unspecified diabetic retinopathy)",
        )

        # Look up ETDRS risk
        etdrs_risk = ETDRS_RISK_MAP.get(
            predicted_stage,
            "Moderate (Clinical monitoring advised)",
        )

        # Build differential diagnosis (sorted by probability)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        differential: List[DiagnosticDifferential] = []
        for stage_name, prob in sorted_probs[:3]:  # Top 3 candidates
            differential.append(
                DiagnosticDifferential(
                    stage=stage_name,
                    probability=round(float(prob), 4),
                    icd10_code=ICD10_MAP.get(stage_name, "E11.319").split()[0],
                )
            )

        # Clinical rationale based on staging
        clean = predicted_stage.lower()
        if "no" in clean:
            rationale = (
                "No microvascular pathology identified. Vasculature exhibits continuous wall margins without focal ectasias. "
                "Baseline annual rescreening recommended under diabetic care protocol."
            )
        elif "mild" in clean:
            rationale = (
                "Meets Early Treatment Diabetic Retinopathy Study (ETDRS) definition for Mild NPDR: isolated microaneurysms only. "
                "Absence of diffuse hemorrhages, hard exudates, or venous changes."
            )
        elif "moderate" in clean:
            rationale = (
                "Meets ETDRS criteria for Moderate NPDR: microaneurysms and intraretinal hemorrhages more extensive than mild NPDR "
                "but less severe than the 4-2-1 threshold. Lipid exudation indicates localized endothelial barrier breakdown."
            )
        elif "severe" in clean:
            rationale = (
                "Meets ETDRS 4-2-1 rule criteria for Severe NPDR: extensive blot hemorrhages in all 4 quadrants, significant venous beading, "
                "and prominent intraretinal microvascular abnormalities (IRMA). Immediate retinal evaluation indicated."
            )
        else:
            rationale = (
                "Proliferative Diabetic Retinopathy diagnosed. Presence of neovascularization, preretinal fibrous proliferation, or "
                "vitreous hemorrhage secondary to retinal hypoxia. Urgent vitreoretinal consultation required."
            )

        return Agent2PathologyOutput(
            primary_icdr_stage=predicted_stage,
            icd10_code=icd10_code,
            conformal_coverage_set=conformal_set,
            etdrs_progression_risk=etdrs_risk,
            differential_diagnosis=differential,
            clinical_rationale=rationale,
        )
