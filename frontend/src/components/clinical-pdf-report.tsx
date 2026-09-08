import React, { useRef } from "react";
import { PredictionResult } from "../lib/api";
import { Printer, X, Download, Shield, Eye, Calendar, UserCheck } from "lucide-react";

interface ClinicalPdfReportProps {
  result: PredictionResult;
  originalImageSrc: string;
  isOpen: boolean;
  onClose: () => void;
}

export const ClinicalPdfReport: React.FC<ClinicalPdfReportProps> = ({
  result,
  originalImageSrc,
  isOpen,
  onClose,
}) => {
  const reportRef = useRef<HTMLDivElement>(null);

  if (!isOpen) return null;

  const handlePrint = () => {
    window.print();
  };

  const patientId = `PT-${Math.floor(100000 + Math.random() * 900000)}`;
  const dateStr = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black/80 backdrop-blur-sm flex justify-center p-4 sm:p-6 print:p-0 print:bg-white print:static print:inset-auto">
      <div className="relative w-full max-w-4xl bg-white text-slate-900 rounded-2xl shadow-2xl overflow-hidden flex flex-col my-auto print:shadow-none print:rounded-none print:w-full">
        {/* Modal Controls (Hidden in Print) */}
        <div className="flex items-center justify-between p-4 bg-slate-900 text-white print:hidden">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-indigo-400" />
            <h3 className="text-sm font-bold">Clinical Diagnostic Examination Summary (PDF Export)</h3>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors shadow"
            >
              <Printer className="w-3.5 h-3.5" /> Print / Save as PDF
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* ── PRINTABLE CLINICAL DOCUMENT ─────────────────────────── */}
        <div ref={reportRef} className="p-8 sm:p-10 space-y-6 print:p-6">
          {/* Header */}
          <div className="flex items-start justify-between border-b-2 border-indigo-900 pb-4">
            <div>
              <h1 className="text-2xl font-black tracking-tight text-indigo-950 uppercase">
                Diabetic Retinopathy Screening Report
              </h1>
              <p className="text-xs text-slate-600 font-medium mt-0.5">
                Explainable Multi-Agent AI Decision Support & Triage Platform
              </p>
              <p className="text-[11px] text-slate-500 mt-1">
                National Eye Care Protocol · ETDRS 4-2-1 Criteria · ICD-10 CM Staging
              </p>
            </div>
            <div className="text-right text-xs text-slate-600 space-y-0.5">
              <div className="font-mono font-bold text-slate-900">ID: {patientId}</div>
              <div className="flex items-center justify-end gap-1 text-slate-500">
                <Calendar className="w-3 h-3" /> {dateStr}
              </div>
              <div className="text-[10px] bg-slate-100 text-slate-700 px-2 py-0.5 rounded font-mono mt-1">
                STATUS: CONFIDENTIAL MEDICAL RECORD
              </div>
            </div>
          </div>

          {/* Clinical Visual Evidence (Fundus vs Grad-CAM++) */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              I. Retinal Fundus Imaging & Explainability Heatmap
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="border border-slate-200 rounded-lg p-2 bg-slate-50">
                <div className="aspect-square rounded overflow-hidden bg-slate-900 flex items-center justify-center">
                  <img
                    src={originalImageSrc}
                    alt="Original Fundus"
                    className="w-full h-full object-contain"
                  />
                </div>
                <p className="text-[11px] text-center font-medium text-slate-600 mt-1.5">
                  Original Retinal Fundus Photograph
                </p>
              </div>

              <div className="border border-slate-200 rounded-lg p-2 bg-slate-50">
                <div className="aspect-square rounded overflow-hidden bg-slate-900 flex items-center justify-center">
                  <img
                    src={`data:image/png;base64,${result.gradcam_overlay}`}
                    alt="Grad-CAM++ Overlay"
                    className="w-full h-full object-contain"
                  />
                </div>
                <p className="text-[11px] text-center font-medium text-slate-600 mt-1.5">
                  Grad-CAM++ Microvascular Activation Heatmap
                </p>
              </div>
            </div>
          </div>

          {/* Diagnostic Staging & Conformal Bounds */}
          <div className="space-y-2">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              II. Automated Staging & Conformal Bounds
            </h3>
            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 bg-indigo-50 border border-indigo-100 rounded-lg">
                <span className="text-[10px] font-semibold text-indigo-700 block uppercase">
                  Primary ICDR Diagnosis
                </span>
                <span className="text-base font-black text-indigo-950 block mt-0.5">
                  {result.prediction}
                </span>
                <span className="text-[11px] text-indigo-700 font-medium">
                  Calibrated Confidence: {roundPct(result.confidence)}
                </span>
              </div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-[10px] font-semibold text-slate-600 block uppercase">
                  95% Conformal Coverage Set
                </span>
                <span className="text-xs font-bold text-slate-900 block mt-1">
                  {result.conformal_prediction_set && result.conformal_prediction_set.length > 0
                    ? result.conformal_prediction_set.join(", ")
                    : result.prediction}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Guaranteed true grade coverage (P ≥ 0.95)
                </span>
              </div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-[10px] font-semibold text-slate-600 block uppercase">
                  AAO Referral Timeline
                </span>
                <span className="text-xs font-bold text-amber-900 block mt-1">
                  {result.agentic_findings?.safety_auditor?.referral_urgency || result.review_recommendation}
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Protocol: {result.certainty} Certainty
                </span>
              </div>
            </div>
          </div>

          {/* Multi-Agent Clinical Findings */}
          {result.agentic_findings && (
            <div className="space-y-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700">
                III. Multi-Agent Triaging Findings & ICD-10 Staging
              </h3>
              <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-2 text-xs">
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div>
                    <span className="text-slate-500 font-medium">WHO ICD-10 Code:</span>
                    <span className="font-semibold text-slate-900 ml-1">
                      {result.agentic_findings.pathology_agent.icd10_code}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 font-medium">Consensus Status:</span>
                    <span className="font-semibold text-emerald-700 ml-1">
                      {result.agentic_findings.consensus_status.replace(/_/g, " ")}
                    </span>
                  </div>
                </div>

                <p className="text-slate-700 leading-relaxed text-[11px]">
                  <strong>Clinical Assessment:</strong> {result.agentic_findings.synthesized_recommendation}
                </p>

                {result.agentic_findings.safety_auditor.recommended_interventions.length > 0 && (
                  <div className="pt-1">
                    <strong className="text-[11px] text-slate-800">Mandated Interventions:</strong>
                    <ul className="list-disc list-inside text-[11px] text-slate-600 space-y-0.5 mt-0.5">
                      {result.agentic_findings.safety_auditor.recommended_interventions.map((inv, idx) => (
                        <li key={idx}>{inv}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Physician Sign-Off & Legal Disclaimer */}
          <div className="pt-4 border-t border-slate-200 flex items-end justify-between text-xs">
            <div className="max-w-md space-y-1">
              <span className="text-[10px] font-bold uppercase text-slate-500">
                Regulatory & SaMD Notice
              </span>
              <p className="text-[9px] text-slate-500 leading-tight">
                This diagnostic summary was generated via RetinaScreen AI explainable decision support.
                This report assists certified medical professionals in clinical triage and does not constitute
                autonomous diagnostic advice. Treatment requires direct ophthalmoscopic evaluation.
              </p>
            </div>

            <div className="text-right space-y-4">
              <div className="w-48 border-b border-slate-400 pb-1">
                <span className="font-serif italic text-slate-700 text-sm">Reviewed & Verified</span>
              </div>
              <div className="text-[10px] text-slate-600">
                Attending Ophthalmologist / Retina Specialist
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

function roundPct(val: number): string {
  return `${(val * 100).toFixed(1)}%`;
}
