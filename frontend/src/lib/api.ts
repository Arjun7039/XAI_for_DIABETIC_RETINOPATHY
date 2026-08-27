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

const API_BASE_URL = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

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
