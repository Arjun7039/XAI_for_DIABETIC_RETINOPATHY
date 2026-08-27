/**
 * API Client for RetinaScreen AI FastAPI Backend
 */

export interface PredictionResult {
  prediction: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  certainty: "HIGH" | "LOW";
  review_recommendation: "Recommended" | "Strongly Recommended";
  image_quality: "good" | "poor";
  gradcam_overlay: string; // base64 PNG
  shap_overlay?: string;    // base64 PNG
  model_version: string;
}

export interface QualityRejectResult {
  image_quality: "poor";
  quality_issues: string[];
  message: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Helper to generate a canvas-based sample heatmap base64 for fallback demo mode
function createSampleHeatmapB64(): string {
  if (typeof window === "undefined") return "";
  const canvas = document.createElement("canvas");
  canvas.width = 384;
  canvas.height = 384;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  // Radial gradient imitating Grad-CAM heatmap
  const grad = ctx.createRadialGradient(192, 192, 20, 192, 192, 160);
  grad.addColorStop(0, "rgba(220, 38, 38, 0.85)");   // Core red hotspot
  grad.addColorStop(0.4, "rgba(234, 179, 8, 0.65)");  // Yellow glow
  grad.addColorStop(0.7, "rgba(14, 165, 233, 0.4)");  // Cyan border
  grad.addColorStop(1, "rgba(15, 23, 42, 0)");        // Transparent edge

  ctx.fillStyle = "rgb(15, 23, 42)";
  ctx.fillRect(0, 0, 384, 384);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 384, 384);

  return canvas.toDataURL("image/png").replace(/^data:image\/png;base64,/, "");
}

export async function analyzeRetinalImage(
  file: File
): Promise<{ data?: PredictionResult; rejection?: QualityRejectResult; error?: string; isDemo?: boolean }> {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 sec timeout for CPU inference & XAI

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.status === 422) {
      const rejectData: QualityRejectResult = await response.json();
      return { rejection: rejectData };
    }

    if (!response.ok) {
      const errorText = await response.text();
      return { error: `Backend server error (${response.status}): ${errorText || response.statusText}` };
    }

    const data: PredictionResult = await response.json();
    return { data };
  } catch (err: any) {
    if (err.name === "AbortError") {
      return { error: "Inference timed out (took longer than 60s). Please try again." };
    }
    return { error: `Failed to connect to backend at ${API_BASE_URL}. Ensure uvicorn server is running.` };
  }
}

