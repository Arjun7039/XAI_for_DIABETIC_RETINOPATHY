"""
Unit tests for the 2026 Agentic Multi-Agent Clinical Triage Engine.
"""

import pytest
from app.agents.triage_agent import ClinicalTriageAgent
from app.agents.pathology_agent import DiagnosticPathologyAgent
from app.agents.safety_auditor_agent import ClinicalSafetyAuditorAgent
from app.agents.orchestrator import clinical_orchestrator


@pytest.mark.asyncio
async def test_triage_agent_detects_proliferative_hallmarks():
    agent = ClinicalTriageAgent()
    res = await agent.analyze(
        predicted_stage="Proliferative_DR",
        confidence=0.92,
        blur_score=150.0,
        contrast_score=45.0,
    )
    assert res.status == "COMPLETED"
    assert res.image_quality_grade == "Optimal"
    assert any("Neovascularization" in l.name for l in res.lesions_detected)
    assert res.macular_involvement is True


@pytest.mark.asyncio
async def test_pathology_agent_assigns_icd10_and_conformal_bounds():
    agent = DiagnosticPathologyAgent()
    res = await agent.stage(
        predicted_stage="Moderate_NPDR",
        probabilities={"Moderate_NPDR": 0.85, "Severe_NPDR": 0.10, "Mild_NPDR": 0.05},
        conformal_set=["Moderate_NPDR"],
    )
    assert "E11.339" in res.icd10_code
    assert "Moderate" in res.etdrs_progression_risk
    assert res.conformal_coverage_set == ["Moderate_NPDR"]
    assert len(res.differential_diagnosis) > 0


@pytest.mark.asyncio
async def test_safety_auditor_enforces_aao_referral_urgency():
    auditor = ClinicalSafetyAuditorAgent()
    triage_agent = ClinicalTriageAgent()
    pathology_agent = DiagnosticPathologyAgent()

    triage_out = await triage_agent.analyze("Proliferative_DR", 0.95, 120.0, 42.0)
    pathology_out = await pathology_agent.stage("Proliferative_DR", {"Proliferative_DR": 0.95}, ["Proliferative_DR"])

    audit_out = await auditor.audit(triage_out, pathology_out, certainty="HIGH", confidence=0.95)
    assert "Emergency" in audit_out.referral_urgency
    assert len(audit_out.recommended_interventions) > 0


@pytest.mark.asyncio
async def test_multi_agent_orchestrator_consensus_flow():
    report = await clinical_orchestrator.run_consultation(
        predicted_stage="Mild_NPDR",
        confidence=0.88,
        probabilities={"Mild_NPDR": 0.88, "No_DR": 0.10, "Moderate_NPDR": 0.02},
        conformal_set=["Mild_NPDR"],
        certainty="HIGH",
        blur_score=110.0,
        contrast_score=38.0,
    )
    assert report.consensus_reached is True
    assert "Agreement" in report.consensus_status.replace("_", " ").title()
    assert report.triage_agent.agent_name.startswith("Agent 1")
    assert report.pathology_agent.agent_name.startswith("Agent 2")
    assert report.safety_auditor.agent_name.startswith("Agent 3")
