import React from "react";
import { MultiAgentConsensusReport } from "../lib/api";
import {
  Brain,
  ShieldCheck,
  Stethoscope,
  AlertTriangle,
  CheckCircle,
  FileBadge,
  Clock,
  Activity,
  Layers,
} from "lucide-react";

interface AgenticConsultationProps {
  findings?: MultiAgentConsensusReport;
}

export const AgenticConsultation: React.FC<AgenticConsultationProps> = ({ findings }) => {
  if (!findings) return null;

  const {
    consensus_reached,
    consensus_status,
    triage_agent,
    pathology_agent,
    safety_auditor,
    synthesized_recommendation,
  } = findings;

  return (
    <div className="w-full bg-slate-900/90 backdrop-blur-md rounded-2xl border border-purple-500/20 p-6 shadow-2xl space-y-6 animate-fade-in-up">
      {/* ── Header & Consensus Status ────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-purple-500/20 text-purple-400 border border-purple-500/30">
            <Brain className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-white tracking-tight">
                2026 Agentic AI Clinical Triage System
              </h3>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-purple-500/20 text-purple-300 border border-purple-400/30">
                Multi-Agent Loop
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Autonomous consultation between 3 specialized clinical agents with protocol guardrails
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {consensus_reached ? (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs font-semibold">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              {consensus_status.replace(/_/g, " ")}
            </div>
          ) : (
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs font-semibold">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              ESCALATED FOR CLINICIAN REVIEW
            </div>
          )}
        </div>
      </div>

      {/* ── Three Specialized Agent Cards ───────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* ── AGENT 1: Clinical Triage Specialist ───────────────── */}
        <div className="bg-slate-950/60 rounded-xl border border-slate-800 p-4 space-y-3 flex flex-col justify-between">
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold uppercase tracking-wider text-blue-400 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5" /> Agent 1: Optical Triage
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/15 text-blue-300 font-medium">
                {triage_agent.image_quality_grade} Clarity
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              {triage_agent.summary}
            </p>

            {/* Quality Metrics */}
            <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] text-slate-400">
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Sharpness Index</span>
                <span className="font-semibold text-slate-200">{triage_agent.blur_score}</span>
              </div>
              <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Optic Disc</span>
                <span className="font-semibold text-slate-200">{triage_agent.optic_disc_visibility}</span>
              </div>
            </div>

            {/* Detected Lesion Hallmarks */}
            {triage_agent.lesions_detected.length > 0 && (
              <div className="space-y-1.5 pt-1">
                <span className="text-[11px] font-semibold text-slate-400 block">
                  ETDRS Lesion Hallmarks:
                </span>
                <div className="space-y-1">
                  {triage_agent.lesions_detected.map((l, i) => (
                    <div
                      key={i}
                      className="p-1.5 rounded bg-blue-950/40 border border-blue-800/40 text-[11px] flex items-center justify-between text-blue-200"
                    >
                      <span className="truncate">{l.name}</span>
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-blue-500/20 text-blue-300 shrink-0 font-medium">
                        {l.quadrants_affected}Q ({l.severity})
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── AGENT 2: Diagnostic & Staging Specialist ──────────── */}
        <div className="bg-slate-950/60 rounded-xl border border-slate-800 p-4 space-y-3 flex flex-col justify-between">
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-1.5">
                <Stethoscope className="w-3.5 h-3.5" /> Agent 2: Pathology Staging
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-300 font-mono font-semibold">
                ICD-10 Staged
              </span>
            </div>

            {/* Primary ICD-10 Code Badge */}
            <div className="p-2 rounded-lg bg-indigo-950/40 border border-indigo-500/30 text-xs">
              <span className="text-slate-400 text-[10px] block">WHO Classification:</span>
              <span className="font-semibold text-indigo-200 text-[11px] block mt-0.5">
                {pathology_agent.icd10_code}
              </span>
            </div>

            {/* Conformal Coverage Set */}
            <div className="p-2 rounded-lg bg-slate-900/80 border border-slate-800 text-xs space-y-1">
              <div className="flex items-center justify-between text-[10px]">
                <span className="text-slate-400 font-medium flex items-center gap-1">
                  <Layers className="w-3 h-3 text-indigo-400" /> 95% Conformal Set:
                </span>
                <span className="text-emerald-400 font-mono font-bold">P ≥ 0.95</span>
              </div>
              <div className="flex flex-wrap gap-1 pt-0.5">
                {pathology_agent.conformal_coverage_set.map((stage, i) => (
                  <span
                    key={i}
                    className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[11px] font-medium border border-indigo-400/20"
                  >
                    {stage}
                  </span>
                ))}
              </div>
            </div>

            {/* Progression Risk */}
            <div className="text-[11px] text-slate-300">
              <span className="text-slate-500 text-[10px] block">1-Year Progression Risk:</span>
              <span className="font-medium text-slate-200">{pathology_agent.etdrs_progression_risk}</span>
            </div>
          </div>
        </div>

        {/* ── AGENT 3: Safety & Protocol Auditor ────────────────── */}
        <div className="bg-slate-950/60 rounded-xl border border-slate-800 p-4 space-y-3 flex flex-col justify-between">
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5" /> Agent 3: Safety Auditor
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 font-semibold">
                AAO Guardrails
              </span>
            </div>

            {/* Referral Urgency Highlight */}
            <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-xs">
              <span className="text-amber-400/90 text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1">
                <Clock className="w-3 h-3" /> Mandatory Referral Protocol:
              </span>
              <span className="text-amber-200 font-bold text-xs block mt-1">
                {safety_auditor.referral_urgency}
              </span>
            </div>

            {/* Safety Checkpoints */}
            <div className="space-y-1 text-[11px]">
              {safety_auditor.safety_checks.map((chk, i) => (
                <div key={i} className="flex items-start gap-1.5 text-slate-300">
                  {chk.passed ? (
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                  ) : (
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                  )}
                  <span className="text-[10px] text-slate-300 leading-tight">
                    <strong className="text-slate-200">{chk.criterion}:</strong> {chk.details}
                  </span>
                </div>
              ))}
            </div>

            {/* Recommended Interventions */}
            {safety_auditor.recommended_interventions.length > 0 && (
              <div className="pt-1">
                <span className="text-[10px] font-semibold text-slate-400 block mb-1">
                  Primary Interventions:
                </span>
                <ul className="text-[10px] text-slate-300 space-y-0.5 list-disc list-inside">
                  {safety_auditor.recommended_interventions.slice(0, 2).map((inv, idx) => (
                    <li key={idx} className="truncate">{inv}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Executive Recommendation Summary ─────────────────────── */}
      <div className="p-4 rounded-xl bg-purple-950/30 border border-purple-500/30 flex items-start gap-3">
        <FileBadge className="w-5 h-5 text-purple-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="text-xs font-bold text-purple-200 uppercase tracking-wide">
            Synthesized Clinical Consultation Summary
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            {synthesized_recommendation}
          </p>
          <p className="text-[10px] text-slate-500 italic pt-1">
            {safety_auditor.regulatory_disclaimer}
          </p>
        </div>
      </div>
    </div>
  );
};
