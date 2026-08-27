import React from "react";
import { BarChart3 } from "lucide-react";

interface ConfidenceBarProps {
  probabilities: Record<string, number>;
  predictedClass: string;
}

const CLASS_COLORS: Record<string, { bar: string; glow: string }> = {
  "No DR":            { bar: "bg-emerald-500", glow: "shadow-[0_0_8px_rgba(16,185,129,0.4)]" },
  "Mild DR":          { bar: "bg-sky-500", glow: "shadow-[0_0_8px_rgba(14,165,233,0.4)]" },
  "Moderate DR":      { bar: "bg-amber-500", glow: "shadow-[0_0_8px_rgba(245,158,11,0.4)]" },
  "Severe DR":        { bar: "bg-orange-500", glow: "shadow-[0_0_8px_rgba(249,115,22,0.4)]" },
  "Proliferative DR": { bar: "bg-rose-500", glow: "shadow-[0_0_8px_rgba(244,63,94,0.4)]" },
  // Also handle config-style class names
  "Mild_NPDR":        { bar: "bg-sky-500", glow: "shadow-[0_0_8px_rgba(14,165,233,0.4)]" },
  "Moderate_NPDR":    { bar: "bg-amber-500", glow: "shadow-[0_0_8px_rgba(245,158,11,0.4)]" },
  "No_DR":            { bar: "bg-emerald-500", glow: "shadow-[0_0_8px_rgba(16,185,129,0.4)]" },
  "Proliferative_DR": { bar: "bg-rose-500", glow: "shadow-[0_0_8px_rgba(244,63,94,0.4)]" },
  "Severe_NPDR":      { bar: "bg-orange-500", glow: "shadow-[0_0_8px_rgba(249,115,22,0.4)]" },
};

const DEFAULT_COLOR = { bar: "bg-indigo-500", glow: "" };

export function ConfidenceBar({ probabilities, predictedClass }: ConfidenceBarProps) {
  const classes = Object.keys(probabilities);

  return (
    <div className="space-y-4">
      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
        <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
        Softmax Probability Distribution
      </h4>
      <div className="space-y-3">
        {classes.map((cls, index) => {
          const prob = probabilities[cls] || 0;
          const percentage = (prob * 100).toFixed(1);
          const isPredicted = cls === predictedClass;
          const colors = CLASS_COLORS[cls] || DEFAULT_COLOR;

          return (
            <div
              key={cls}
              className="space-y-1.5"
              style={{ animationDelay: `${index * 80}ms` }}
            >
              <div className="flex justify-between text-xs font-medium">
                <span className={isPredicted ? "text-slate-100 font-bold flex items-center gap-2" : "text-slate-400"}>
                  {cls.replace(/_NPDR|_DR|_/g, " ").trim()}
                  {isPredicted && (
                    <span className="text-[10px] px-2 py-0.5 rounded-md bg-indigo-500/15 text-indigo-300 border border-indigo-500/25 font-bold">
                      ▸ Predicted
                    </span>
                  )}
                </span>
                <span className={isPredicted ? "text-indigo-300 font-extrabold" : "text-slate-500"}>
                  {percentage}%
                </span>
              </div>
              <div className="h-2.5 w-full bg-slate-800/60 rounded-full overflow-hidden border border-slate-700/30">
                <div
                  className={`h-full rounded-full animate-bar-fill ${
                    isPredicted ? `${colors.bar} ${colors.glow}` : "bg-slate-700/50"
                  }`}
                  style={{ width: `${Math.max(parseFloat(percentage), 0.5)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
