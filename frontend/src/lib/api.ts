/**
 * API Client for RetinaScreen AI FastAPI Backend
 * Includes 2026 Multi-Agent Clinical Findings, Conformal Prediction, Presets, and SSE Streaming.
 */

export interface LesionHallmark {
  name: string;
  confidence: number;
  severity: "Mild" | "Moderate" | "Severe";
  quadrants_affected: number;
  clinical_note: string;
}

export interface Agent1TriageOutput {
  agent_name: string;
  status: string;
  image_quality_grade: "Optimal" | "Diagnostic" | "Suboptimal";
  blur_score: number;
  contrast_score: number;
  lesions_detected: LesionHallmark[];
  optic_disc_visibility: string;
  macular_involvement: boolean;
  summary: string;
}

export interface DiagnosticDifferential {
  stage: string;
  probability: number;
  icd10_code: string;
}

export interface Agent2PathologyOutput {
  agent_name: string;
  status: string;
  primary_icdr_stage: string;
  icd10_code: string;
  conformal_coverage_set: string[];
  etdrs_progression_risk: string;
  differential_diagnosis: DiagnosticDifferential[];
  clinical_rationale: string;
}

export interface SafetyCheck {
  criterion: string;
  passed: boolean;
  details: string;
}

export interface Agent3AuditorOutput {
  agent_name: string;
  status: string;
  referral_urgency: string;
  recommended_interventions: string[];
  safety_checks: SafetyCheck[];
  contraindications_or_warnings: string[];
  regulatory_disclaimer: string;
}

export interface MultiAgentConsensusReport {
  consensus_reached: boolean;
  consensus_status: "UNANIMOUS_AGREEMENT" | "MAJORITY_STAGED" | "ESCALATED_FOR_MANUAL_REVIEW";
  triage_agent: Agent1TriageOutput;
  pathology_agent: Agent2PathologyOutput;
  safety_auditor: Agent3AuditorOutput;
  synthesized_recommendation: string;
}

export interface PredictionResult {
  prediction: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  certainty: "HIGH" | "LOW";
  review_recommendation: "Recommended" | "Strongly Recommended";
  conformal_prediction_set?: string[];
  conformal_coverage?: number;
  image_quality: "good" | "poor";
  gradcam_overlay: string;  // base64 PNG
  saliency_overlay?: string; // base64 PNG
  shap_overlay?: string;     // base64 PNG
  model_version: string;
  agentic_findings?: MultiAgentConsensusReport;
  execution_telemetry_ms?: Record<string, number>;
}

export interface QualityRejectResult {
  image_quality: "poor";
  quality_issues: string[];
  message: string;
}

export interface PresetCase {
  id: string;
  title: string;
  stage_label: string;
  badge_color: string;
  description: string;
  clinical_findings: string;
  thumbnail_b64: string;
}

const RAW_API_URL = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";
export const API_BASE_URL = RAW_API_URL.replace(/\/+$/, "");

/**
 * Fetch recruiter test presets from backend
 */
export async function fetchPresets(): Promise<PresetCase[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/presets`);
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn("Could not load presets from backend, using client-side fallback.", err);
  }
  return [];
}

/**
 * Converts a base64 PNG data string into a standard File object
 */
export function b64ToFile(b64Data: string, filename: string = "preset_retina.png"): File {
  const byteString = atob(b64Data.replace(/^data:image\/\w+;base64,/, ""));
  const arrayBuffer = new ArrayBuffer(byteString.length);
  const uint8Array = new Uint8Array(arrayBuffer);
  for (let i = 0; i < byteString.length; i++) {
    uint8Array[i] = byteString.charCodeAt(i);
  }
  const blob = new Blob([uint8Array], { type: "image/png" });
  return new File([blob], filename, { type: "image/png" });
}

export async function analyzeRetinalImage(
  file: File,
  fastTriage: boolean = true
): Promise<{ data?: PredictionResult; rejection?: QualityRejectResult; error?: string }> {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 120 sec timeout for free-tier cloud CPU cold-starts

    const response = await fetch(`${API_BASE_URL}/predict?fast_triage=${fastTriage}`, {
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
      return { error: "Inference timed out (cloud server cold-start took longer than 120s). Please try submitting again." };
    }
    return { error: `Failed to connect to backend at ${API_BASE_URL}. If deploying on Render Free Tier, the instance may be waking up from spin-down. Please retry in 15-20 seconds.` };
  }
}


