import React from "react";
import { Eye, ShieldAlert } from "lucide-react";
import { API_BASE_URL } from "../lib/api";

export function Header() {
  return (
    <header className="border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-xl sticky top-0 z-50 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-gradient-to-br from-indigo-600/25 to-purple-600/20 border border-indigo-500/25 rounded-xl text-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.15)]">
            <Eye className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-extrabold flex items-center gap-2.5">
              <span className="bg-gradient-to-r from-white via-slate-100 to-indigo-200 bg-clip-text text-transparent">
                RetinaScreen AI
              </span>
              <span className="text-[10px] font-bold px-2.5 py-0.5 rounded-full bg-gradient-to-r from-indigo-500/15 to-purple-500/10 text-indigo-300 border border-indigo-500/20 tracking-wide">
                v1.0 ICDR
              </span>
            </h1>
            <p className="text-xs text-slate-400/80">
              Explainable & Uncertainty-Aware Diabetic Retinopathy Screening
            </p>
          </div>
        </div>

        {/* Clinical Disclaimer & Endpoint badge */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 text-xs text-slate-400 bg-slate-900/60 px-3 py-1.5 rounded-xl border border-slate-800/60">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
            <span className="font-mono text-[11px] text-slate-300">{API_BASE_URL}</span>
          </div>

          <div className="flex items-center gap-2 bg-amber-500/10 border border-amber-500/20 text-amber-300/90 text-xs px-3.5 py-1.5 rounded-xl max-w-xl backdrop-blur-sm">
            <ShieldAlert className="w-4 h-4 shrink-0 text-amber-400" />
            <span>
              <strong>Clinical Screening Aid:</strong> Requires confirmation by an ophthalmologist.
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
