import React, { useEffect, useState } from "react";
import { fetchPresets, PresetCase, b64ToFile } from "../lib/api";
import { Sparkles, ArrowRight, ShieldAlert, CheckCircle2, Eye } from "lucide-react";

interface RecruiterPresetsProps {
  onSelectPreset: (file: File) => void;
  isLoading: boolean;
}

export const RecruiterPresets: React.FC<RecruiterPresetsProps> = ({
  onSelectPreset,
  isLoading,
}) => {
  const [presets, setPresets] = useState<PresetCase[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    fetchPresets().then((data) => {
      if (data && data.length > 0) {
        setPresets(data);
      }
    });
  }, []);

  const handleSelect = (preset: PresetCase) => {
    if (isLoading) return;
    setActiveId(preset.id);
    const file = b64ToFile(preset.thumbnail_b64, `${preset.id}.png`);
    onSelectPreset(file);
  };

  if (presets.length === 0) return null;

  return (
    <div className="w-full bg-slate-900/80 backdrop-blur-md rounded-2xl border border-indigo-500/20 p-5 shadow-xl space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/20 text-indigo-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide uppercase flex items-center gap-2">
              Recruiter & Clinician Evaluation Presets
              <span className="text-[10px] normal-case bg-indigo-500/20 text-indigo-300 font-normal px-2 py-0.5 rounded-full border border-indigo-400/30">
                1-Click Instant Demo
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Test automated 5-class grading, conformal coverage bounds, and multi-agent reasoning instantly.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {presets.map((p) => {
          const isSelected = activeId === p.id;
          const isReject = p.stage_label.toLowerCase().includes("reject");

          return (
            <button
              key={p.id}
              onClick={() => handleSelect(p)}
              disabled={isLoading}
              className={`text-left group relative p-3 rounded-xl border transition-all duration-200 flex flex-col justify-between ${
                isSelected
                  ? "bg-indigo-950/60 border-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.25)] ring-1 ring-indigo-400"
                  : "bg-slate-800/50 border-slate-700/60 hover:bg-slate-800 hover:border-slate-600"
              } ${isLoading ? "opacity-60 cursor-not-allowed" : "cursor-pointer"}`}
            >
              <div className="space-y-2 w-full">
                <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-slate-950 border border-slate-700/50 flex items-center justify-center">
                  <img
                    src={`data:image/png;base64,${p.thumbnail_b64}`}
                    alt={p.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute top-1.5 right-1.5">
                    {isReject ? (
                      <span className="px-1.5 py-0.5 rounded bg-amber-500/90 text-slate-950 text-[10px] font-bold flex items-center gap-1">
                        <ShieldAlert className="w-2.5 h-2.5" /> Reject
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded bg-emerald-500/90 text-slate-950 text-[10px] font-bold flex items-center gap-1">
                        <CheckCircle2 className="w-2.5 h-2.5" /> {p.stage_label}
                      </span>
                    )}
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-slate-200 line-clamp-1 group-hover:text-indigo-300 transition-colors">
                    {p.title}
                  </h4>
                  <p className="text-[11px] text-slate-400 line-clamp-2 mt-0.5 leading-snug">
                    {p.description}
                  </p>
                </div>
              </div>

              <div className="pt-2 mt-2 border-t border-slate-700/40 flex items-center justify-between w-full text-[11px] font-medium text-indigo-400 group-hover:text-indigo-300">
                <span>Evaluate Stage</span>
                <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
