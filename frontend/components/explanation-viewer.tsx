"use client";

import React, { useState } from "react";
import { Layers, Flame, Sparkles, Info, Maximize2 } from "lucide-react";

interface ExplanationViewerProps {
  originalImageSrc: string;
  gradcamOverlayB64: string;
  shapOverlayB64?: string;
}

export function ExplanationViewer({
  originalImageSrc,
  gradcamOverlayB64,
  shapOverlayB64,
}: ExplanationViewerProps) {
  const [activeTab, setActiveTab] = useState<"gradcam" | "shap" | "original">("gradcam");

  const gradcamSrc = `data:image/png;base64,${gradcamOverlayB64}`;
  const shapSrc = shapOverlayB64 ? `data:image/png;base64,${shapOverlayB64}` : null;

  const tabConfig = [
    {
      key: "gradcam" as const,
      label: "Grad-CAM",
      icon: Flame,
      show: true,
      activeColor: "bg-gradient-to-r from-indigo-600 to-indigo-500 shadow-lg shadow-indigo-600/30",
    },
    {
      key: "shap" as const,
      label: "Saliency",
      icon: Sparkles,
      show: !!shapSrc,
      activeColor: "bg-gradient-to-r from-purple-600 to-purple-500 shadow-lg shadow-purple-600/30",
    },
    {
      key: "original" as const,
      label: "Original",
      icon: Maximize2,
      show: true,
      activeColor: "bg-slate-700 shadow-lg shadow-slate-700/30",
    },
  ];

  const descriptions: Record<string, React.ReactNode> = {
    gradcam: (
      <span>
        <strong className="text-slate-200">Grad-CAM:</strong>{" "}
        Highlights spatial regions in the final convolutional layer that strongly influenced
        the severity score — look for hotspots on exudates, hemorrhages, and neovascularization.
      </span>
    ),
    shap: (
      <span>
        <strong className="text-slate-200">Gradient Saliency:</strong>{" "}
        Pixel-level gradient attribution map showing which input regions had the strongest
        influence on the model prediction. Brighter = higher influence.
      </span>
    ),
    original: (
      <span>
        <strong className="text-slate-200">Original Image:</strong>{" "}
        Clean reference fundus photograph as uploaded, before any explainability overlays.
      </span>
    ),
  };

  return (
    <div className="glass-card rounded-2xl p-5 space-y-4">
      {/* Header & Tab Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/50 pb-3">
        <h3 className="text-sm font-bold text-slate-200 flex items-center gap-2.5">
          <div className="p-1.5 bg-indigo-500/15 rounded-lg border border-indigo-500/20">
            <Layers className="w-4 h-4 text-indigo-400" />
          </div>
          Explainable AI (XAI) Overlays
        </h3>

        {/* Tab Toggle */}
        <div className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800/60 self-start sm:self-auto">
          {tabConfig
            .filter((t) => t.show)
            .map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all duration-300 flex items-center gap-1.5 ${
                  activeTab === tab.key
                    ? `${tab.activeColor} text-white`
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <tab.icon className="w-3.5 h-3.5" />
                {tab.label}
              </button>
            ))}
        </div>
      </div>

      {/* Main Image Display */}
      <div className="relative aspect-square w-full max-h-[400px] bg-black/60 rounded-xl overflow-hidden border border-slate-800/50 flex items-center justify-center">
        {activeTab === "gradcam" && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={gradcamSrc}
            alt="Grad-CAM Heatmap Overlay"
            className="object-contain w-full h-full animate-fade-in-up"
          />
        )}
        {activeTab === "shap" && shapSrc && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={shapSrc}
            alt="Gradient Saliency Attribution Map"
            className="object-contain w-full h-full animate-fade-in-up"
          />
        )}
        {activeTab === "original" && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={originalImageSrc}
            alt="Original Retinal Photograph"
            className="object-contain w-full h-full animate-fade-in-up"
          />
        )}

        {/* Overlay label */}
        <div className="absolute top-3 left-3 px-2.5 py-1 rounded-lg bg-black/60 backdrop-blur-sm border border-white/10 text-[10px] font-bold text-white/80 uppercase tracking-wider">
          {activeTab === "gradcam" ? "Grad-CAM" : activeTab === "shap" ? "Saliency" : "Original"}
        </div>
      </div>

      {/* Context Description */}
      <div className="bg-slate-900/50 rounded-xl p-3.5 border border-slate-800/40 text-xs text-slate-400/90 flex items-start gap-2.5 leading-relaxed">
        <Info className="w-4 h-4 text-indigo-400/70 shrink-0 mt-0.5" />
        <div>{descriptions[activeTab]}</div>
      </div>
    </div>
  );
}
