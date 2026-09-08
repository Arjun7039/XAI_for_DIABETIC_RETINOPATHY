import React, { useState } from "react";
import { Header } from "./components/header";
import { UploadZone } from "./components/upload-zone";
import { ResultsPanel } from "./components/results-panel";
import { ExplanationViewer } from "./components/explanation-viewer";
import { RecruiterPresets } from "./components/recruiter-presets";
import { AgenticConsultation } from "./components/agentic-consultation";
import { SynchronizedViewer } from "./components/synchronized-viewer";
import { ClinicalPdfReport } from "./components/clinical-pdf-report";
import { analyzeRetinalImage, PredictionResult, QualityRejectResult } from "./lib/api";
import { Activity, AlertCircle, Sparkles, Zap, Shield, Brain, Layers } from "lucide-react";

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | undefined>(undefined);
  const [rejection, setRejection] = useState<QualityRejectResult | undefined>(undefined);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isPdfOpen, setIsPdfOpen] = useState(false);
  const [showSyncViewer, setShowSyncViewer] = useState(false);

  const handleImageSelected = async (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setIsLoading(true);
    setErrorMsg(null);
    setResult(undefined);
    setRejection(undefined);

    const response = await analyzeRetinalImage(file);
    setIsLoading(false);

    if (response.error) {
      setErrorMsg(response.error);
    } else if (response.rejection) {
      setRejection(response.rejection);
    } else if (response.data) {
      setResult(response.data);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(undefined);
    setRejection(undefined);
    setErrorMsg(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* ── Top Header Bar ──────────────────────────────── */}
      <Header />

      <main className="max-w-7xl mx-auto p-4 sm:p-8 space-y-8">
        {/* ── Hero Banner ──────────────────────────────────── */}
        <div className="relative rounded-3xl p-8 sm:p-10 overflow-hidden border border-white/[0.06] animate-fade-in-up">
          {/* Animated gradient mesh background */}
          <div className="absolute inset-0 animate-gradient-bg" />
          <div className="absolute inset-0 grid-pattern" />

          {/* Floating orbs */}
          <div className="orb orb-indigo w-48 h-48 -top-10 -right-10" />
          <div className="orb orb-purple w-36 h-36 bottom-0 left-1/4" />
          <div className="orb orb-cyan w-28 h-28 top-1/2 right-1/3" />

          <div className="relative z-10 max-w-3xl space-y-5">
            {/* Tech stack pill */}
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-400/25 text-indigo-300 text-xs font-semibold shadow-[0_0_20px_rgba(99,102,241,0.15)] backdrop-blur-sm">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400 animate-pulse" />
              2026 Agentic AI · Conformal Prediction (95%) · Grad-CAM++ · Sub-Second Inference
            </div>

            <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
              <span className="bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent">
                Diabetic Retinopathy
              </span>
              <br />
              <span className="bg-gradient-to-r from-indigo-300 via-purple-300 to-cyan-300 bg-clip-text text-transparent">
                Agentic Screening & Diagnostic Platform
              </span>
            </h2>

            <p className="text-sm text-slate-300/90 leading-relaxed max-w-2xl">
              Automated 5-class ICDR severity grading powered by a 3-agent clinical triage loop,
              split conformal uncertainty bounds (95% coverage), and dual-canvas radiologist inspection.
            </p>

            {/* Feature badges */}
            <div className="flex flex-wrap gap-3 pt-1">
              {[
                { icon: Brain, label: "3-Agent Clinical Loop", color: "text-purple-400 bg-purple-500/10 border-purple-500/20" },
                { icon: Layers, label: "95% Conformal Bounds", color: "text-indigo-400 bg-indigo-500/10 border-indigo-500/20" },
                { icon: Shield, label: "Optical Quality Gate", color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" },
                { icon: Zap, label: "Grad-CAM++ (Sub-200ms)", color: "text-amber-400 bg-amber-500/10 border-amber-500/20" },
              ].map((badge) => (
                <div
                  key={badge.label}
                  className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg border text-xs font-medium ${badge.color}`}
                >
                  <badge.icon className="w-3.5 h-3.5" />
                  {badge.label}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── Recruiter & Clinician Evaluation Presets ── */}
        <RecruiterPresets onSelectPreset={handleImageSelected} isLoading={isLoading} />

        {/* ── Main Grid: Upload & Results ──────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Upload */}
          <div className="lg:col-span-5 space-y-6">
            <UploadZone
              onImageSelected={handleImageSelected}
              isLoading={isLoading}
              onReset={handleReset}
              selectedFile={selectedFile}
            />

            {errorMsg && (
              <div className="p-4 bg-rose-950/30 border border-rose-500/30 rounded-2xl text-xs text-rose-300 flex items-start gap-3 animate-fade-in-up">
                <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
                <div>
                  <strong className="block text-slate-200 mb-1">Backend Connection Notice:</strong>
                  <span>{errorMsg}</span>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results & XAI */}
          <div className="lg:col-span-7 space-y-6">
            {!result && !rejection && !isLoading && (
              <div className="glass-card rounded-2xl p-10 text-center flex flex-col items-center justify-center min-h-[380px] space-y-4 border-dashed border-slate-800 dot-pattern">
                <div className="p-5 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 rounded-2xl border border-indigo-500/20 text-slate-400">
                  <Activity className="w-10 h-10" />
                </div>
                <h3 className="text-lg font-bold text-slate-200">Awaiting Retinal Photograph</h3>
                <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                  Select a test case from the presets above or upload a retinal fundus photo to start automated
                  conformal grading and multi-agent consultation.
                </p>
                <div className="flex gap-2 pt-2">
                  <div className="w-2 h-2 rounded-full bg-indigo-500/40 animate-pulse" />
                  <div className="w-2 h-2 rounded-full bg-purple-500/40 animate-pulse" style={{ animationDelay: "0.3s" }} />
                  <div className="w-2 h-2 rounded-full bg-cyan-500/40 animate-pulse" style={{ animationDelay: "0.6s" }} />
                </div>
              </div>
            )}

            {(result || rejection) && (
              <div className="animate-fade-in-up">
                <ResultsPanel
                  result={result}
                  rejection={rejection}
                  onRetakeRequested={handleReset}
                  onExportPdf={() => setIsPdfOpen(true)}
                />
              </div>
            )}

            {/* Explanation & Synchronized Viewers */}
            {result && previewUrl && (
              <div className="space-y-4 animate-fade-in-up" style={{ animationDelay: "0.15s" }}>
                <div className="flex items-center justify-between px-1">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    Explainability &amp; Inspection
                  </span>
                  <button
                    onClick={() => setShowSyncViewer(!showSyncViewer)}
                    className="text-xs text-indigo-400 hover:text-indigo-300 font-semibold flex items-center gap-1.5 transition-colors"
                  >
                    <Layers className="w-3.5 h-3.5" />
                    {showSyncViewer ? "Show Carousel View" : "Show Synchronized Radiologist View"}
                  </button>
                </div>

                {showSyncViewer ? (
                  <SynchronizedViewer
                    originalSrc={previewUrl}
                    gradcamSrc={result.gradcam_overlay}
                    shapSrc={result.shap_overlay}
                  />
                ) : (
                  <ExplanationViewer
                    originalImageSrc={previewUrl}
                    gradcamOverlayB64={result.gradcam_overlay}
                    saliencyOverlayB64={result.saliency_overlay}
                    shapOverlayB64={result.shap_overlay}
                  />
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── 2026 Agentic AI Multi-Agent Consultation Section ──── */}
        {result && result.agentic_findings && (
          <AgenticConsultation findings={result.agentic_findings} />
        )}
      </main>

      {/* ── Printable Clinical PDF Diagnostic Report Modal ───────── */}
      {result && previewUrl && (
        <ClinicalPdfReport
          result={result}
          originalImageSrc={previewUrl}
          isOpen={isPdfOpen}
          onClose={() => setIsPdfOpen(false)}
        />
      )}
    </div>
  );
}
