"""
Multi-Agent Orchestrator for RetinaScreen AI.
Coordinates Agent 1 (Triage), Agent 2 (Pathology), and Agent 3 (Safety Auditor) in an asynchronous consensus graph.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from app.agents.pathology_agent import DiagnosticPathologyAgent
from app.agents.safety_auditor_agent import ClinicalSafetyAuditorAgent
from app.agents.schemas import MultiAgentConsensusReport
from app.agents.triage_agent import ClinicalTriageAgent


class ClinicalMultiAgentOrchestrator:
    """
    Coordinates asynchronous execution of specialized medical agents:
    Step 1: Clinical Triage Agent (Visual analysis & lesion extraction)
    Step 2: Diagnostic Pathology Agent (Staging, ICD-10, ETDRS risk)
    Step 3: Safety & Referral Auditor Agent (Protocol guardrails & AAO referral urgency)
    Step 4: Consensus resolution & executive summary synthesis
    """

    def __init__(self):
        self.triage_agent = ClinicalTriageAgent()
        self.pathology_agent = DiagnosticPathologyAgent()
        self.safety_auditor = ClinicalSafetyAuditorAgent()

    async def run_consultation(
        self,
        predicted_stage: str,
        confidence: float,
        probabilities: Dict[str, float],
        conformal_set: List[str],
        certainty: str,
        blur_score: float,
        contrast_score: float,
        gradcam_stats: Optional[Dict[str, float]] = None,
    ) -> MultiAgentConsensusReport:
        """
        Runs the full 3-agent clinical consultation pipeline.
        """
        # Step 1 & Step 2 can run concurrently
        triage_task = asyncio.create_task(
            self.triage_agent.analyze(
                predicted_stage=predicted_stage,
                confidence=confidence,
                blur_score=blur_score,
                contrast_score=contrast_score,
                gradcam_stats=gradcam_stats,
            )
        )
        pathology_task = asyncio.create_task(
            self.pathology_agent.stage(
                predicted_stage=predicted_stage,
                probabilities=probabilities,
                conformal_set=conformal_set,
            )
        )

        triage_output, pathology_output = await asyncio.gather(triage_task, pathology_task)

        # Step 3: Safety Auditor runs with insights from both Triage and Pathology
        auditor_output = await self.safety_auditor.audit(
            triage_output=triage_output,
            pathology_output=pathology_output,
            certainty=certainty,
            confidence=confidence,
        )

        # Step 4: Consensus Evaluation
        has_warnings = len(auditor_output.contraindications_or_warnings) > 0
        all_checks_passed = all(c.passed for c in auditor_output.safety_checks)

        if not has_warnings and all_checks_passed:
            consensus_status = "UNANIMOUS_AGREEMENT"
            consensus_reached = True
        elif not has_warnings:
            consensus_status = "MAJORITY_STAGED"
            consensus_reached = True
        else:
            consensus_status = "ESCALATED_FOR_MANUAL_REVIEW"
            consensus_reached = False

        # Build executive summary
        summary = (
            f"Automated clinical triaging completed with {consensus_status.replace('_', ' ').title()}. "
            f"Primary diagnostic classification is {pathology_output.primary_icdr_stage} ({pathology_output.icd10_code.split()[0]}) "
            f"with calibrated confidence of {round(confidence * 100, 1)}% and 95% conformal bounds spanning {len(conformal_set)} stage(s). "
            f"Clinical protocol mandates {auditor_output.referral_urgency}."
        )

        return MultiAgentConsensusReport(
            consensus_reached=consensus_reached,
            consensus_status=consensus_status,
            triage_agent=triage_output,
            pathology_agent=pathology_output,
            safety_auditor=auditor_output,
            synthesized_recommendation=summary,
        )


# Global singleton instance
clinical_orchestrator = ClinicalMultiAgentOrchestrator()
