import React, { useRef, useState } from "react";
import { UploadCloud, RefreshCw, FileImage } from "lucide-react";

interface UploadZoneProps {
  onImageSelected: (file: File) => void;
  isLoading: boolean;
  onReset: () => void;
  selectedFile: File | null;
}

export function UploadZone({ onImageSelected, isLoading, onReset, selectedFile }: UploadZoneProps) {
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const handleFile = (file: File) => {
    if (file && file.type.startsWith("image/")) {
      setPreviewUrl(URL.createObjectURL(file));
      onImageSelected(file);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleClear = () => {
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = "";
    onReset();
  };

  return (
    <div className="w-full glass-card rounded-2xl p-6 transition-all duration-300">
      <h2 className="text-lg font-bold text-slate-100 mb-1.5 flex items-center gap-2.5">
        <div className="p-1.5 bg-indigo-500/15 rounded-lg border border-indigo-500/20">
          <FileImage className="w-4 h-4 text-indigo-400" />
        </div>
        Retinal Fundus Input
      </h2>
      <p className="text-xs text-slate-400/80 mb-5 pl-9">
        Upload a high-resolution macula- or optic-disc-centered fundus photograph
      </p>

      {!previewUrl ? (
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center min-h-[280px] group overflow-hidden ${
            dragActive
              ? "border-indigo-400 bg-indigo-500/10 scale-[0.98] shadow-[0_0_40px_rgba(99,102,241,0.25)]"
              : "border-slate-700/60 hover:border-indigo-500/50 hover:bg-slate-800/30"
          }`}
        >
          {/* Animated background glow on hover */}
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/[0.04] via-transparent to-purple-500/[0.04] opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />
          
          {/* Dot pattern */}
          <div className="absolute inset-0 dot-pattern opacity-30 pointer-events-none" />

          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="hidden"
          />

          <div className={`relative p-5 rounded-2xl mb-4 border transition-all duration-500 ${
            dragActive
              ? "bg-indigo-500/20 border-indigo-400/40 shadow-[0_0_30px_rgba(99,102,241,0.3)]"
              : "bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border-indigo-500/15 group-hover:border-indigo-400/30 group-hover:shadow-[0_0_25px_rgba(99,102,241,0.15)]"
          }`}>
            <UploadCloud className={`w-9 h-9 transition-all duration-300 ${
              dragActive ? "text-indigo-300 scale-110" : "text-indigo-400/70 group-hover:text-indigo-300"
            }`} />
          </div>

          <p className="text-sm font-semibold text-slate-200 mb-1 relative z-10">
            {dragActive ? "Release to upload" : "Drag and drop fundus image here"}
          </p>
          {!dragActive && (
            <p className="text-xs text-slate-400/80 relative z-10">
              or{" "}
              <span className="text-indigo-400 font-semibold underline underline-offset-2 decoration-indigo-400/40 hover:decoration-indigo-400">
                browse files
              </span>
            </p>
          )}
          <p className="text-[11px] text-slate-500/60 mt-3 relative z-10">
            JPEG, PNG up to 20MB · Quality check runs automatically
          </p>
        </div>
      ) : (
        <div className="relative rounded-2xl overflow-hidden border border-slate-700/40 bg-slate-900/60 p-4 animate-fade-in-up">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs text-slate-300 truncate max-w-[200px] font-medium flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              {selectedFile?.name || "Fundus Image"}
            </span>
            <button
              onClick={handleClear}
              disabled={isLoading}
              className="text-xs text-slate-400 hover:text-white flex items-center gap-1.5 bg-slate-800/80 hover:bg-slate-700/80 px-3 py-1.5 rounded-lg border border-slate-700/60 transition-all duration-200"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              New Image
            </button>
          </div>

          <div className="relative aspect-square max-h-[340px] w-full mx-auto rounded-xl overflow-hidden border border-slate-800/60 bg-black flex items-center justify-center">
            <img
              src={previewUrl}
              alt="Uploaded Fundus Preview"
              className="object-contain max-h-full w-full"
            />
            {isLoading && (
              <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-sm flex flex-col items-center justify-center space-y-4">
                {/* Scan line */}
                <div className="scan-line" />
                
                <div className="relative">
                  <div className="w-14 h-14 border-[3px] border-indigo-500/30 rounded-full" />
                  <div className="absolute inset-0 w-14 h-14 border-[3px] border-indigo-400 border-t-transparent rounded-full animate-spin" />
                </div>
                <div className="text-center space-y-1">
                  <p className="text-sm text-indigo-300 font-semibold">
                    Analyzing...
                  </p>
                  <p className="text-[11px] text-slate-400/80 animate-pulse">
                    Quality gate → EfficientNet-B4 inference → XAI overlays
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
