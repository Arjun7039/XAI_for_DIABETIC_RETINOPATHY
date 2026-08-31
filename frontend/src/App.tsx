import React, { useState } from "react";
import { Header } from "./components/header";
import { UploadZone } from "./components/upload-zone";
import { ResultsPanel } from "./components/results-panel";
import { ExplanationViewer } from "./components/explanation-viewer";
import { analyzeRetinalImage, PredictionResult, QualityRejectResult } from "./lib/api";
import { Activity, AlertCircle, Sparkles, Zap, Shield, Brain } from "lucide-react";

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | undefined>(undefined);
  const [rejection, setRejection] = useState<QualityRejectResult | undefined>(undefined);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [isDemo, setIsDemo] = useState(false);

  const handleImageSelected = async (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setIsLoading(true);
    setErrorMsg(null);
    setResult(undefined);
    setRejection(undefined);
    setIsDemo(false);

    const response = await analyzeRetinalImage(file);
    setIsLoading(false);

    if (response.isDemo) {
      setIsDemo(true);
    }

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
            EfficientNet-B4 · Grad-CAM · Saliency · SHAP
          </div>

          <h2 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
            <span className="bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent">
              Diabetic Retinopathy
            </span>
            <br />
            <span className="bg-gradient-to-r from-indigo-300 via-purple-300 to-cyan-300 bg-clip-text text-transparent">
              Screening & Diagnostic AI
            </span>
          </h2>

          <p className="text-sm text-slate-300/90 leading-relaxed max-w-2xl">
            Upload fundus photographs for automated 5-class ICDR severity grading with
            image-quality verification, calibrated confidence scoring, and dual
            explainability overlays powered by deep learning.
          </p>

          {/* Feature badges */}
          <div className="flex flex-wrap gap-3 pt-1">
            {[
              { icon: Zap, label: "5-Class ICDR", color: "text-amber-400 bg-amber-500/10 border-amber-500/20" },
              { icon: Shield, label: "Quality Gate", color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" },
              { icon: Brain, label: "Explainable AI", color: "text-purple-400 bg-purple-500/10 border-purple-500/20" },
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
                <strong className="block text-slate-200 mb-1">Backend Connection Note:</strong>
                <span>{errorMsg}</span>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Results & XAI */}
        <div className="lg:col-span-7 space-y-6">
          {isDemo && (
            <div className="p-3 bg-sky-950/40 border border-sky-500/30 rounded-xl text-xs text-sky-300 flex items-center justify-between animate-fade-in-up">
              <span className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-sky-400" />
                <strong>Local UI Preview Mode:</strong> FastAPI backend offline.
              </span>
            </div>
          )}

          {!result && !rejection && !isLoading && (
            <div className="glass-card rounded-2xl p-10 text-center flex flex-col items-center justify-center min-h-[380px] space-y-4 border-dashed border-slate-800 dot-pattern">
              <div className="p-5 bg-gradient-to-br from-indigo-500/10 to-purple-500/10 rounded-2xl border border-indigo-500/20 text-slate-400">
                <Activity className="w-10 h-10" />
              </div>
              <h3 className="text-lg font-bold text-slate-200">Awaiting Image Input</h3>
              <p className="text-xs text-slate-500 max-w-sm leading-relaxed">
                Select or drag a retinal fundus photograph on the left to begin
                quality evaluation and severity classification.
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
              />
            </div>
          )}

          {result && previewUrl && (
            <div className="animate-fade-in-up" style={{ animationDelay: "0.15s" }}>
              <ExplanationViewer
                originalImageSrc={previewUrl}
                gradcamOverlayB64={result.gradcam_overlay}
                saliencyOverlayB64={result.saliency_overlay}
                shapOverlayB64={result.shap_overlay}
              />
            </div>
          )}
        </div>
      </div>
      </main>
    </div>
  );
}
