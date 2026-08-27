"use client";

import React from "react";
import { PredictionResult, QualityRejectResult } from "@/lib/api";
import { ConfidenceBar } from "./confidence-bar";
import { ShieldCheck, AlertTriangle, Stethoscope, RefreshCw, TrendingUp } from "lucide-react";

interface ResultsPanelProps {
  result?: PredictionResult;
  rejection?: QualityRejectResult;
  onRetakeRequested: () => void;
}

// Model Index Mapping (Alphabetical from Kaggle image_dataset_from_directory):
// 0: Mild_NPDR
// 1: Moderate_NPDR
// 2: No_DR
// 3: Proliferative_DR
// 4: Severe_NPDR

const SEVERITY_CONFIG: Record<number, { label: string; color: string; glow: string; bgGradient: string; gradeText: string }> = {
  2: {
    label: "No DR",
    gradeText: "Grade 0 / 4",
    color: "text-emerald-300",
    glow: "shadow-[0_0_20px_rgba(16,185,129,0.25)]",
    bgGradient: "from-emerald-500/15 to-emerald-500/5",
  },
  0: {
    label: "Mild",
    gradeText: "Grade 1 / 4",
    color: "text-sky-300",
    glow: "shadow-[0_0_20px_rgba(14,165,233,0.25)]",
    bgGradient: "from-sky-500/15 to-sky-500/5",
  },
  1: {
    label: "Moderate",
    gradeText: "Grade 2 / 4",
    color: "text-amber-300",
    glow: "shadow-[0_0_20px_rgba(245,158,11,0.25)]",
    bgGradient: "from-amber-500/15 to-amber-500/5",
  },
  4: {
    label: "Severe",
    gradeText: "Grade 3 / 4",
    color: "text-orange-300",
    glow: "shadow-[0_0_20px_rgba(249,115,22,0.3)]",
    bgGradient: "from-orange-500/15 to-orange-500/5",
  },
  3: {
    label: "Proliferative",
    gradeText: "Grade 4 / 4",
    color: "text-rose-300",
    glow: "shadow-[0_0_20px_rgba(244,63,94,0.3)]",
    bgGradient: "from-rose-500/15 to-rose-500/5",
  },
};

export function ResultsPanel({ result, rejection, onRetakeRequested }: ResultsPanelProps) {
  if (rejection) {
    return (
      <div className="glass-card rounded-2xl p-6 border-amber-500/25 space-y-4 animate-fade-in-up">
        <div className="flex items-center gap-3 text-amber-400">
          <div className="p-2.5 bg-amber-500/15 rounded-xl border border-amber-500/25">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">Image Quality Check Failed</h3>
            <p className="text-xs text-amber-300/70">Quality gate triggered — inference skipped</p>
          </div>
        </div>

        <div className="p-4 bg-slate-900/70 rounded-xl border border-slate-800/60 text-xs space-y-2.5">
          <p className="text-slate-300 font-medium">{rejection.message}</p>
          <div className="flex flex-wrap gap-2 pt-1">
            <span className="text-slate-400 font-semibold">Detected Issues:</span>
            {rejection.quality_issues.map((issue) => (
              <span
                key={issue}
                className="px-2.5 py-0.5 rounded-lg bg-rose-500/15 text-rose-300 border border-rose-500/25 font-mono text-[11px]"
              >
                {issue}
              </span>
            ))}
          </div>
        </div>

        <button
          onClick={onRetakeRequested}
          className="w-full py-3 px-4 bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-white text-xs font-bold rounded-xl transition-all duration-300 flex items-center justify-center gap-2 shadow-lg shadow-amber-600/20 hover:shadow-amber-500/30"
        >
          <RefreshCw className="w-4 h-4" />
          Retake Fundus Photograph
        </button>
      </div>
    );
  }

  if (!result) return null;

  const isHighCertainty = result.certainty === "HIGH";
  const isSevere = result.class_index >= 3;
  const severity = SEVERITY_CONFIG[result.class_index] || SEVERITY_CONFIG[0];

  return (
    <div className="glass-card rounded-2xl p-6 space-y-6 shimmer-effect">
      {/* ── Top Prediction Box ──────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/60 pb-5">
        <div className="space-y-2">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-widest">
            ICDR Severity Grade
          </span>
          <div className="flex items-center gap-3">
            <h2 className={`text-2xl font-extrabold ${severity.color}`}>
              {result.prediction.replace(/_NPDR|_DR|_/g, " ").trim()} {result.prediction.includes("No_DR") || result.prediction === "No DR" ? "Retinopathy (No DR)" : "DR"}
            </h2>
            <span
              className={`text-xs px-3 py-1.5 rounded-xl border font-bold bg-gradient-to-r ${severity.bgGradient} border-white/10 ${severity.glow}`}
            >
              {severity.gradeText}
            </span>
          </div>
        </div>

        {/* Confidence & Certainty Badges */}
        <div className="flex flex-wrap sm:flex-col items-start sm:items-end gap-2.5">
          <div className="flex items-center gap-2 bg-slate-900/70 px-3.5 py-2 rounded-xl border border-slate-800/60">
            <TrendingUp className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-xs text-slate-400">Confidence:</span>
            <span className="text-sm font-extrabold bg-gradient-to-r from-indigo-300 to-purple-300 bg-clip-text text-transparent">
              {(result.confidence * 100).toFixed(1)}%
            </span>
          </div>

          <div
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-bold border ${
              isHighCertainty
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25 shadow-[0_0_12px_rgba(16,185,129,0.15)]"
                : "bg-amber-500/10 text-amber-400 border-amber-500/25 animate-pulse shadow-[0_0_12px_rgba(245,158,11,0.2)]"
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            Certainty: {result.certainty}
          </div>
        </div>
      </div>

      {/* ── Review Recommendation Banner ────────────────── */}
      <div
        className={`p-4 rounded-xl border flex items-start gap-3 ${
          isSevere || !isHighCertainty
            ? "bg-rose-950/20 border-rose-500/30 text-rose-200 animate-glow-red"
            : "bg-indigo-950/20 border-indigo-500/20 text-indigo-200"
        }`}
      >
        <Stethoscope className={`w-5 h-5 shrink-0 mt-0.5 ${
          isSevere || !isHighCertainty ? "text-rose-400" : "text-indigo-400"
        }`} />
        <div className="space-y-1">
          <h4 className="text-xs font-bold uppercase tracking-wide text-slate-100">
            Screening Recommendation: {result.review_recommendation}
          </h4>
          <p className="text-xs text-slate-300/80 leading-relaxed">
            {result.review_recommendation === "Strongly Recommended"
              ? "Flagged for priority ophthalmic triage. Low model certainty or elevated severe stage detected."
              : "Standard clinical follow-up recommended per diabetic eye screening guidelines."}
          </p>
        </div>
      </div>

      {/* ── Probability Distribution ───────────────────── */}
      <ConfidenceBar
        probabilities={result.probabilities}
        predictedClass={result.prediction}
      />
    </div>
  );
}
