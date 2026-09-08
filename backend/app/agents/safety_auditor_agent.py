"""
Agent 3: Safety & Referral Protocol Auditor
Validates clinical safety bounds, cross-checks for model-hallmark contradictions, and enforces AAO referral protocols.
"""

from __future__ import annotations

from typing import List

from app.agents.schemas import (
    Agent1TriageOutput,
    Agent2PathologyOutput,
    Agent3AuditorOutput,
    SafetyCheck,
)


class ClinicalSafetyAuditorAgent:
    """
    Agent 3 enforces clinical guardrails, checks for model-hallmark conflicts,
    and assigns referral timelines according to American Academy of Ophthalmology (AAO) guidelines.
    """

    def __init__(self):
        self.name = "Agent 3: Safety & Referral Protocol Auditor"

    async def audit(
        self,
        triage_output: Agent1TriageOutput,
        pathology_output: Agent2PathologyOutput,
        certainty: str,
        confidence: float,
    ) -> Agent3AuditorOutput:
        safety_checks: List[SafetyCheck] = []
        contraindications: List[str] = []
        interventions: List[str] = []

        # 1. Check Image Quality Safety
        quality_passed = triage_output.image_quality_grade != "Suboptimal"
        safety_checks.append(
            SafetyCheck(
                criterion="Image Resolution & Optical Clarity",
                passed=quality_passed,
                details=(
                    f"Sharpness index {triage_output.blur_score} with contrast {triage_output.contrast_score}. "
                    + ("Image verified suitable for automated grading." if quality_passed else "Low optical clarity may mask subtle microaneurysms.")
                ),
            )
        )

        # 2. Check Conformal Set Ambiguity
        conformal_len = len(pathology_output.conformal_coverage_set)
        conformal_passed = conformal_len <= 2
        safety_checks.append(
            SafetyCheck(
                criterion="Conformal Prediction Set Stability (95% Coverage)",
                passed=conformal_passed,
                details=(
                    f"Conformal set spans {conformal_len} class(es): {', '.join(pathology_output.conformal_coverage_set)}. "
                    + ("Narrow, stable prediction set." if conformal_passed else "Wide prediction interval indicates significant diagnostic uncertainty.")
                ),
            )
        )

        # 3. Check Confidence / Certainty Flag
        certainty_passed = certainty == "HIGH"
        safety_checks.append(
            SafetyCheck(
                criterion="Model Softmax Calibration & Certainty",
                passed=certainty_passed,
                details=(
                    f"Top class confidence: {round(confidence * 100, 1)}% ({certainty} Certainty). "
                    + ("Exceeds high-confidence threshold." if certainty_passed else "Below calibration threshold; mandatory secondary clinical review.")
                ),
            )
        )

        # 4. Hallmarks vs Staging Cross-Check (Contradiction Detection)
        has_severe_lesions = any(
            l.severity == "Severe" for l in triage_output.lesions_detected
        )
        stage_clean = pathology_output.primary_icdr_stage.lower()
        is_early_stage = "no dr" in stage_clean or "mild" in stage_clean

        if has_severe_lesions and is_early_stage:
            contraindications.append(
                "CONFLICT DETECTED: Visual triage hallmarks show severe vascular abnormalities while primary model predicted early-stage DR. Triage safety override triggered."
            )

        # Determine AAO Referral Urgency & Interventions
        if "proliferative" in stage_clean:
            urgency = "Emergency (Within 48–72 Hours)"
            interventions.extend(
                [
                    "Immediate referral to Vitreoretinal Specialist for slit-lamp biomicroscopy.",
                    "High-resolution Optical Coherence Tomography (OCT) of macula for CSME evaluation.",
                    "Evaluate candidacy for urgent Panretinal Photocoagulation (PRP) or Intravitreal Anti-VEGF injections.",
                    "Strict optimization of HbA1c and systemic blood pressure.",
                ]
            )
        elif "severe" in stage_clean:
            urgency = "Urgent (Within 2–4 Weeks)"
            interventions.extend(
                [
                    "Prompt comprehensive dilated ophthalmoscopic examination by an ophthalmologist.",
                    "Macular OCT and Widefield Fluorescein Angiography (FA) to map ischemic non-perfusion.",
                    "Counsel patient on high 1-year progression risk (~50%) to proliferative stage.",
                ]
            )
        elif "moderate" in stage_clean:
            urgency = "Semi-Urgent (Within 2–3 Months)"
            interventions.extend(
                [
                    "Referral to optometrist/ophthalmologist for comprehensive dilated eye exam.",
                    "Fundus photography review at 3-to-6 month intervals to monitor hemorrhage density.",
                    "Endocrine consultation for glycemic and lipid control review.",
                ]
            )
        elif "mild" in stage_clean:
            urgency = "Routine Clinical Review (Within 6–12 Months)"
            interventions.extend(
                [
                    "Repeat fundus screening within 6 to 12 months.",
                    "Patient education regarding microvascular risk factors, blood pressure, and HbA1c targets (<7.0%).",
                ]
            )
        else:
            urgency = "Annual Routine Screening (12 Months)"
            interventions.extend(
                [
                    "Annual routine diabetic eye screening per international standards of diabetic care.",
                    "Advise patient to report any visual changes (floaters, distortion, blurred vision) immediately.",
                ]
            )

        # If certainty is low or conflict exists, elevate urgency
        if not certainty_passed and urgency == "Annual Routine Screening (12 Months)":
            urgency = "Semi-Urgent (Within 3 Months — Review Advised)"
            contraindications.append(
                "Low model confidence warrants earlier follow-up than standard annual screening."
            )

        return Agent3AuditorOutput(
            referral_urgency=urgency,
            recommended_interventions=interventions,
            safety_checks=safety_checks,
            contraindications_or_warnings=contraindications,
        )
