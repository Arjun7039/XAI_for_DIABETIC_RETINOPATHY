import React, { useState } from "react";
import { Sliders, Maximize2, ZoomIn, ZoomOut, RotateCcw, Layers } from "lucide-react";

interface SynchronizedViewerProps {
  originalSrc: string;
  gradcamSrc: string;
  shapSrc?: string;
}

export const SynchronizedViewer: React.FC<SynchronizedViewerProps> = ({
  originalSrc,
  gradcamSrc,
  shapSrc,
}) => {
  const [zoom, setZoom] = useState(1);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [activeXai, setActiveXai] = useState<"gradcam" | "shap">("gradcam");
  const [viewMode, setViewMode] = useState<"side-by-side" | "blend">("side-by-side");

  const currentOverlay = activeXai === "gradcam" ? gradcamSrc : shapSrc || gradcamSrc;

  const handleZoomIn = () => setZoom((z) => Math.min(2.5, z + 0.25));
  const handleZoomOut = () => setZoom((z) => Math.max(1, z - 0.25));
  const handleReset = () => {
    setZoom(1);
    setOverlayOpacity(0.5);
  };

  return (
    <div className="w-full bg-slate-900/80 backdrop-blur-md rounded-2xl border border-slate-800 p-5 shadow-xl space-y-4">
      {/* Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-white">
            Synchronized Dual-Canvas Radiologist Inspector
          </h4>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {/* Mode Switcher */}
          <div className="inline-flex rounded-lg bg-slate-800 p-0.5 border border-slate-700">
            <button
              onClick={() => setViewMode("side-by-side")}
              className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${
                viewMode === "side-by-side"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setViewMode("blend")}
              className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${
                viewMode === "blend"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Blend Slider
            </button>
          </div>

          {/* Active Overlay Switcher */}
          {shapSrc && (
            <div className="inline-flex rounded-lg bg-slate-800 p-0.5 border border-slate-700">
              <button
                onClick={() => setActiveXai("gradcam")}
                className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                  activeXai === "gradcam"
                    ? "bg-purple-600 text-white"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Grad-CAM++
              </button>
              <button
                onClick={() => setActiveXai("shap")}
                className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                  activeXai === "shap"
                    ? "bg-purple-600 text-white"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                KernelSHAP
              </button>
            </div>
          )}

          {/* Zoom Buttons */}
          <div className="flex items-center gap-1 bg-slate-800 px-1 py-0.5 rounded-lg border border-slate-700 text-slate-300">
            <button onClick={handleZoomOut} className="p-1 hover:text-white" title="Zoom Out">
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] font-mono px-1 font-semibold">{Math.round(zoom * 100)}%</span>
            <button onClick={handleZoomIn} className="p-1 hover:text-white" title="Zoom In">
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button onClick={handleReset} className="p-1 hover:text-white ml-1" title="Reset Zoom">
              <RotateCcw className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Blend Slider if in blend mode */}
      {viewMode === "blend" && (
        <div className="flex items-center gap-3 bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 text-xs">
          <Sliders className="w-4 h-4 text-slate-400" />
          <span className="text-slate-400 text-[11px]">Heatmap Opacity:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
            className="w-full accent-indigo-500 cursor-pointer"
          />
          <span className="text-[11px] font-mono font-bold text-indigo-300 w-10 text-right">
            {Math.round(overlayOpacity * 100)}%
          </span>
        </div>
      )}

      {/* Synchronized Display Area */}
      {viewMode === "side-by-side" ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Left Canvas: Raw Fundus */}
          <div className="space-y-2">
            <span className="text-[11px] font-semibold text-slate-400 block text-center">
              Target Retinal Field (High-Res)
            </span>
            <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
              <div
                style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}
                className="w-full h-full transition-transform duration-200 flex items-center justify-center"
              >
                <img
                  src={originalSrc}
                  alt="Original fundus"
                  className="w-full h-full object-contain pointer-events-none"
                />
              </div>
            </div>
          </div>

          {/* Right Canvas: Synced XAI Heatmap */}
          <div className="space-y-2">
            <span className="text-[11px] font-semibold text-purple-400 block text-center">
              Synchronized {activeXai === "gradcam" ? "Grad-CAM++" : "SHAP"} Attributions
            </span>
            <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-slate-950 border border-purple-500/30 flex items-center justify-center">
              <div
                style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}
                className="w-full h-full transition-transform duration-200 flex items-center justify-center"
              >
                <img
                  src={`data:image/png;base64,${currentOverlay}`}
                  alt="XAI Overlay"
                  className="w-full h-full object-contain pointer-events-none"
                />
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Single Canvas: Interactive Blended View */
        <div className="relative aspect-square max-w-xl mx-auto rounded-2xl overflow-hidden bg-slate-950 border border-indigo-500/30">
          <div
            style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}
            className="w-full h-full relative transition-transform duration-200 flex items-center justify-center"
          >
            {/* Base Image */}
            <img
              src={originalSrc}
              alt="Base fundus"
              className="absolute inset-0 w-full h-full object-contain"
            />
            {/* Heatmap Overlay with Opacity */}
            <img
              src={`data:image/png;base64,${currentOverlay}`}
              alt="Heatmap overlay"
              style={{ opacity: overlayOpacity }}
              className="absolute inset-0 w-full h-full object-contain transition-opacity"
            />
          </div>
        </div>
      )}
    </div>
  );
};
