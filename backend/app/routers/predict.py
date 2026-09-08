"""
Prediction router for RetinaScreen AI.

Endpoint:
  POST /predict — multipart image upload
"""

from __future__ import annotations

import cv2
import numpy as np
import base64
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

import time
import json
import asyncio
from fastapi.responses import JSONResponse, StreamingResponse

from app.agents import clinical_orchestrator
from app.models.gradcam import generate_gradcam_overlay
from app.models.inference import run_inference, run_batch_inference
from app.models.saliency_explainer import generate_saliency_overlay
from app.models.shap_explainer import generate_shap_overlay
from app.preprocessing.quality_check import check_image_quality, get_image_quality_metrics
from app.preprocessing.retinal_preprocessing import preprocess_for_inference
from app.schemas.prediction import (
    PredictionResponse,
    QualityRejectResponse,
    PatientTriageCard,
    BatchTriageResponse,
)

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse, "description": "Successful DR grading, explainability and multi-agent consultation response"},
        422: {"model": QualityRejectResponse, "description": "Image quality gate failed — retake recommended"},
    },
)
async def predict_retinopathy(
    request: Request,
    file: UploadFile = File(...),
    fast_triage: bool = True,
):
    """
    Accepts fundus photograph image file upload, checks image quality,
    runs 5-class DR classification with conformal coverage, generates Grad-CAM++ overlays,
    and executes the 2026 Agentic Multi-Agent clinical consultation.
    When fast_triage=True (default), skips intensive secondary perturbation explainers for sub-second hospital triage.
    """
    start_total = time.perf_counter()

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File uploaded must be a valid image format (JPEG/PNG).",
        )

    # Read image contents into OpenCV BGR numpy array
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Invalid image file contents")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image file: {e}",
        )

    # 1. Quality Gate Check & Metrics
    t_q_start = time.perf_counter()
    quality_metrics = get_image_quality_metrics(image_bgr)
    t_q_ms = (time.perf_counter() - t_q_start) * 1000

    if not quality_metrics["passed"]:
        issues = quality_metrics["issues"]
        if "non_retinal_image" in issues:
            msg = "Uploaded image does not appear to be a retinal fundus photograph (document/non-retinal photo detected)."
        else:
            msg = "Image quality insufficient for reliable grading. Please retake photo."

        reject_payload = QualityRejectResponse(
            image_quality="poor",
            quality_issues=issues,
            message=msg,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=reject_payload.model_dump(),
        )

    # Retrieve shared model and config instance from app state
    model = getattr(request.app.state, "model", None)
    config = getattr(request.app.state, "config", {})
    class_names = getattr(
        request.app.state,
        "class_names",
        ["Mild_NPDR", "Moderate_NPDR", "No_DR", "Proliferative_DR", "Severe_NPDR"],
    )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not initialized on the server.",
        )

    try:
        # 2. Preprocessing
        use_clahe = config.get("use_clahe", False)
        img_size_config = config.get("image_size", 224)
        if isinstance(img_size_config, (list, tuple)):
            img_size = int(img_size_config[0])
        else:
            img_size = int(img_size_config)

        tensor = preprocess_for_inference(image_bgr, use_clahe=use_clahe, img_size=img_size)

        # 3. Model Inference & Conformal Prediction
        t_inf_start = time.perf_counter()
        (
            prediction,
            class_index,
            confidence,
            probabilities,
            certainty,
            review_recommendation,
            conformal_set,
        ) = run_inference(model, tensor, class_names=class_names)
        t_inf_ms = (time.perf_counter() - t_inf_start) * 1000

        # 4. Generate Explainability Overlays (Grad-CAM++ is primary clinical explainer)
        t_xai_start = time.perf_counter()
        gradcam_res = generate_gradcam_overlay(
            model, tensor, image_bgr, target_category=class_index, return_stats=True
        )
        if isinstance(gradcam_res, tuple):
            gradcam_b64, gradcam_stats = gradcam_res
        else:
            gradcam_b64, gradcam_stats = gradcam_res, {}

        if fast_triage:
            saliency_b64 = None
            shap_b64 = None
        else:
            saliency_b64 = generate_saliency_overlay(
                model, tensor, image_bgr, target_category=class_index
            )
            shap_b64 = generate_shap_overlay(
                model, tensor, image_bgr, target_category=class_index
            )
        t_xai_ms = (time.perf_counter() - t_xai_start) * 1000

        # 5. Agentic AI Multi-Agent Clinical Triage Engine
        t_agent_start = time.perf_counter()
        agentic_findings = await clinical_orchestrator.run_consultation(
            predicted_stage=prediction,
            confidence=confidence,
            probabilities=probabilities,
            conformal_set=conformal_set,
            certainty=certainty,
            blur_score=quality_metrics["blur_score"],
            contrast_score=quality_metrics["contrast_score"],
            gradcam_stats=gradcam_stats,
        )
        t_agent_ms = (time.perf_counter() - t_agent_start) * 1000

        t_total_ms = (time.perf_counter() - start_total) * 1000
        telemetry = {
            "quality_check_ms": round(t_q_ms, 1),
            "inference_ms": round(t_inf_ms, 1),
            "xai_generation_ms": round(t_xai_ms, 1),
            "multi_agent_consult_ms": round(t_agent_ms, 1),
            "total_latency_ms": round(t_total_ms, 1),
        }

        # 6. Construct & Return Response
        response = PredictionResponse(
            prediction=prediction,
            class_index=class_index,
            confidence=confidence,
            probabilities=probabilities,
            certainty=certainty,
            review_recommendation=review_recommendation,
            conformal_prediction_set=conformal_set,
            conformal_coverage=0.95,
            image_quality="good",
            gradcam_overlay=gradcam_b64,
            saliency_overlay=saliency_b64,
            shap_overlay=shap_b64,
            model_version=config.get("model_name", "efficientnet_b4"),
            agentic_findings=agentic_findings,
            execution_telemetry_ms=telemetry,
        )

        return response
    except Exception as e:
        print(f"[ERROR] Inference or postprocessing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference process failed: {str(e)}",
        )


@router.post("/predict/stream")
async def predict_stream(
    request: Request,
    file: UploadFile = File(...),
    fast_triage: bool = True,
):
    """
    Server-Sent Events (SSE) streaming endpoint:
    Streams step-by-step progress from quality verification to agentic audit.
    When fast_triage=True (default), generates primary Grad-CAM++ and bypasses secondary perturbations for instant streaming.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File uploaded must be a valid image format (JPEG/PNG).",
        )

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image file.",
        )

    model = getattr(request.app.state, "model", None)
    config = getattr(request.app.state, "config", {})
    class_names = getattr(
        request.app.state,
        "class_names",
        ["Mild_NPDR", "Moderate_NPDR", "No_DR", "Proliferative_DR", "Severe_NPDR"],
    )

    async def event_generator():
        # Step 1: Quality Gate
        yield f"data: {json.dumps({'event': 'progress', 'step': 'quality_gate', 'message': 'Running image quality guardrails...'})}\n\n"
        await asyncio.sleep(0.01)
        quality_metrics = get_image_quality_metrics(image_bgr)

        if not quality_metrics["passed"]:
            yield f"data: {json.dumps({'event': 'rejected', 'quality_issues': quality_metrics['issues'], 'message': 'Quality check failed.'})}\n\n"
            return

        yield f"data: {json.dumps({'event': 'step_complete', 'step': 'quality_gate', 'metrics': quality_metrics})}\n\n"

        # Step 2: Preprocessing
        yield f"data: {json.dumps({'event': 'progress', 'step': 'preprocessing', 'message': 'Standardizing fundus photograph & CLAHE contrast...'})}\n\n"
        await asyncio.sleep(0.01)
        use_clahe = config.get("use_clahe", False)
        img_size_config = config.get("image_size", 224)
        img_size = int(img_size_config[0] if isinstance(img_size_config, (list, tuple)) else img_size_config)
        tensor = preprocess_for_inference(image_bgr, use_clahe=use_clahe, img_size=img_size)

        # Step 3: Model Inference & Conformal Prediction
        yield f"data: {json.dumps({'event': 'progress', 'step': 'inference', 'message': 'Executing deep ensemble forward pass with Split Conformal Bounds...'})}\n\n"
        await asyncio.sleep(0.01)
        (
            prediction,
            class_index,
            confidence,
            probabilities,
            certainty,
            review_recommendation,
            conformal_set,
        ) = run_inference(model, tensor, class_names=class_names)

        yield f"data: {json.dumps({'event': 'step_complete', 'step': 'inference', 'prediction': prediction, 'confidence': confidence, 'conformal_set': conformal_set})}\n\n"

        # Step 4: XAI Generation
        yield f"data: {json.dumps({'event': 'progress', 'step': 'xai', 'message': 'Synthesizing Grad-CAM++ lesion spatial attributions...'})}\n\n"
        await asyncio.sleep(0.01)
        gradcam_res = generate_gradcam_overlay(
            model, tensor, image_bgr, target_category=class_index, return_stats=True
        )
        gradcam_b64, gradcam_stats = (gradcam_res if isinstance(gradcam_res, tuple) else (gradcam_res, {}))
        if fast_triage:
            saliency_b64 = None
            shap_b64 = None
        else:
            saliency_b64 = generate_saliency_overlay(model, tensor, image_bgr, target_category=class_index)
            shap_b64 = generate_shap_overlay(model, tensor, image_bgr, target_category=class_index)

        # Step 5: Agentic Consultation
        yield f"data: {json.dumps({'event': 'progress', 'step': 'agentic_consultation', 'message': 'Running 3-Agent clinical deliberation (Triage, Pathology, Safety Auditor)...'})}\n\n"
        await asyncio.sleep(0.01)
        agentic_findings = await clinical_orchestrator.run_consultation(
            predicted_stage=prediction,
            confidence=confidence,
            probabilities=probabilities,
            conformal_set=conformal_set,
            certainty=certainty,
            blur_score=quality_metrics["blur_score"],
            contrast_score=quality_metrics["contrast_score"],
            gradcam_stats=gradcam_stats,
        )

        final_response = PredictionResponse(
            prediction=prediction,
            class_index=class_index,
            confidence=confidence,
            probabilities=probabilities,
            certainty=certainty,
            review_recommendation=review_recommendation,
            conformal_prediction_set=conformal_set,
            conformal_coverage=0.95,
            image_quality="good",
            gradcam_overlay=gradcam_b64,
            saliency_overlay=saliency_b64,
            shap_overlay=shap_b64,
            model_version=config.get("model_name", "efficientnet_b4"),
            agentic_findings=agentic_findings,
        )

        yield f"data: {json.dumps({'event': 'final_payload', 'data': final_response.model_dump()})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Hospital Multi-Patient Rapid Queue Triage Endpoints ──────────

_PRIORITY_CONFIG = {
    4: {
        "priority": "P1_CRITICAL",
        "label": "P1 - Immediate Referral (<24h)",
        "color": "red",
        "urgency": "Immediate referral to vitreo-retina specialist within 24-48 hours",
        "action": "Urgent panretinal photocoagulation (PRP) or anti-VEGF evaluation",
        "icd10": "E11.359",
    },
    3: {
        "priority": "P1_CRITICAL",
        "label": "P1 - High Risk Pre-Proliferative (<1 wk)",
        "color": "red",
        "urgency": "Referral to medical retina specialist within 1 week (4-2-1 rule met)",
        "action": "Fluorescein angiography & wide-field fundus monitoring",
        "icd10": "E11.349",
    },
    2: {
        "priority": "P2_URGENT",
        "label": "P2 - Urgent Specialist Consult (2-4 wks)",
        "color": "amber",
        "urgency": "Schedule comprehensive dilated ophthalmologic exam within 2 to 4 weeks",
        "action": "Macular OCT evaluation for sub-clinical center-involving diabetic macular edema",
        "icd10": "E11.339",
    },
    1: {
        "priority": "P3_ROUTINE",
        "label": "P3 - Routine Annual Monitoring (6-12 mo)",
        "color": "green",
        "urgency": "Routine dilated retinal examination in 6 to 12 months",
        "action": "Intensive glycemic (HbA1c < 7.0%) and systemic blood pressure management",
        "icd10": "E11.329",
    },
    0: {
        "priority": "P3_ROUTINE",
        "label": "P3 - Annual Diabetic Screening (12 mo)",
        "color": "green",
        "urgency": "Standard annual diabetic retinopathy rescreening in 12 months",
        "action": "Continue primary care diabetes management and patient lifestyle education",
        "icd10": "E11.319",
    },
}


@router.post(
    "/triage/queue",
    response_model=BatchTriageResponse,
    summary="Vectorized multi-patient hospital triage queue",
    description="Processes multiple patient retinal fundus images in a single vectorized CPU pass. Automatically triages, stratifies by clinical urgency, and ranks patients with critical vision-threatening pathology at the top of the queue.",
)
async def triage_patient_queue(
    request: Request,
    files: list[UploadFile] = File(...),
):
    """
    Sub-second hospital triage processor:
    - Preprocesses multiple patient scans concurrently
    - Executes vectorized forward pass via graph-compiled @tf.function
    - Generates 95% conformal prediction validity sets
    - Prioritizes queue: P1 Critical (Red) -> P2 Urgent (Amber) -> P3 Routine (Green)
    """
    start_total = time.perf_counter()
    model = getattr(request.app.state, "model", None)
    config = getattr(request.app.state, "config", {})
    class_names = getattr(
        request.app.state,
        "class_names",
        ["Mild_NPDR", "Moderate_NPDR", "No_DR", "Proliferative_DR", "Severe_NPDR"],
    )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference model is not initialized.",
        )

    img_size_config = config.get("image_size", 224)
    img_size = int(img_size_config[0] if isinstance(img_size_config, (list, tuple)) else img_size_config)

    tensors = []
    patient_metadata = []

    # 1. Concurrently decode & preprocess images
    for idx, f in enumerate(files):
        try:
            raw_bytes = await f.read()
            arr = np.frombuffer(raw_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            # Generate lightweight base64 thumbnail for queue card preview
            thumb = cv2.resize(bgr, (64, 64))
            _, thumb_buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_buf).decode("utf-8")

            # Tensor preprocessing
            t = preprocess_for_inference(bgr, use_clahe=False, img_size=img_size)
            tensors.append(t)

            filename_stem = f.filename.rsplit(".", 1)[0] if f.filename else f"Patient-{idx+1:03d}"
            patient_metadata.append({
                "patient_id": f"PT-{100 + idx}",
                "patient_name": filename_stem.replace("_", " ").title(),
                "thumbnail_b64": thumb_b64,
            })
        except Exception as e:
            print(f"[WARN] Error decoding queue item {idx}: {e}")
            continue

    if not tensors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid retinal images were provided in the queue request.",
        )

    # 2. Vectorized forward pass
    t_inf_start = time.perf_counter()
    raw_results = run_batch_inference(model, tensors, class_names=class_names)
    batch_inf_ms = (time.perf_counter() - t_inf_start) * 1000

    # 3. Assemble Patient Triage Cards & Priority Stratification
    cards: list[PatientTriageCard] = []
    p1_count = 0
    p2_count = 0
    p3_count = 0

    for i, res in enumerate(raw_results):
        meta = patient_metadata[i]
        c_idx = res["class_index"]
        cfg = _PRIORITY_CONFIG.get(c_idx, _PRIORITY_CONFIG[0])

        if cfg["priority"] == "P1_CRITICAL":
            p1_count += 1
        elif cfg["priority"] == "P2_URGENT":
            p2_count += 1
        else:
            p3_count += 1

        card = PatientTriageCard(
            patient_id=meta["patient_id"],
            patient_name=meta["patient_name"],
            prediction=res["prediction"],
            class_index=c_idx,
            confidence=res["confidence"],
            triage_priority=cfg["priority"],
            priority_label=cfg["label"],
            priority_color=cfg["color"],
            referral_urgency=cfg["urgency"],
            recommended_action=cfg["action"],
            conformal_set=res["conformal_set"],
            icd10_code=cfg["icd10"],
            latency_ms=round(batch_inf_ms / len(raw_results), 1),
            thumbnail_b64=meta["thumbnail_b64"],
        )
        cards.append(card)

    # 4. Sort Queue strictly by clinical severity:
    # Priority sorting order: P1_CRITICAL -> P2_URGENT -> P3_ROUTINE
    # Within tier: highest confidence first
    priority_order = {"P1_CRITICAL": 0, "P2_URGENT": 1, "P3_ROUTINE": 2}
    cards.sort(key=lambda c: (priority_order.get(c.triage_priority, 9), -c.confidence))

    total_latency_ms = round((time.perf_counter() - start_total) * 1000, 1)

    return BatchTriageResponse(
        total_patients=len(cards),
        critical_p1_count=p1_count,
        urgent_p2_count=p2_count,
        routine_p3_count=p3_count,
        total_latency_ms=total_latency_ms,
        avg_latency_ms=round(total_latency_ms / len(cards), 1),
        patients=cards,
    )


@router.get(
    "/triage/demo-queue",
    response_model=BatchTriageResponse,
    summary="Simulated hospital triage queue for instant evaluation",
    description="Loads a simulated hospital outpatient triage queue with 5 diverse patients (Proliferative DR, Severe NPDR, Moderate NPDR, Mild NPDR, and Normal) to demonstrate real-time queue prioritization.",
)
async def get_demo_triage_queue(request: Request):
    """
    Demonstrates instant hospital queue triaging without requiring manual file uploads.
    """
    from app.routers.presets import _generate_synthetic_fundus

    cases = [
        ("PT-104", "Elena Rostova (OS)", "proliferative"),
        ("PT-101", "Robert Chen (OD)", "moderate"),
        ("PT-105", "Carlos Mendez (OD)", "severe"),
        ("PT-102", "Sarah Lin (OD)", "no_dr"),
        ("PT-103", "David Kumar (OS)", "mild"),
    ]

    model = getattr(request.app.state, "model", None)
    config = getattr(request.app.state, "config", {})
    class_names = getattr(
        request.app.state,
        "class_names",
        ["Mild_NPDR", "Moderate_NPDR", "No_DR", "Proliferative_DR", "Severe_NPDR"],
    )

    t0 = time.perf_counter()
    tensors = []
    metadata = []

    for pid, name, c_type in cases:
        b64 = _generate_synthetic_fundus(c_type)
        raw = base64.b64decode(b64)
        nparr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        thumb = cv2.resize(bgr, (64, 64))
        _, thumb_buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(thumb_buf).decode("utf-8")

        tensor = preprocess_for_inference(bgr, use_clahe=False, img_size=224)
        tensors.append(tensor)
        metadata.append({
            "patient_id": pid,
            "patient_name": name,
            "thumbnail_b64": thumb_b64,
            "case_type": c_type,
            "full_b64": "data:image/png;base64," + b64,
        })

    if model is not None:
        raw_results = run_batch_inference(model, tensors, class_names=class_names)
    else:
        raw_results = []

    cards: list[PatientTriageCard] = []
    p1 = 0
    p2 = 0
    p3 = 0

    for i, meta in enumerate(metadata):
        if raw_results and i < len(raw_results):
            c_idx = raw_results[i]["class_index"]
            pred = raw_results[i]["prediction"]
            conf = raw_results[i]["confidence"]
            c_set = raw_results[i]["conformal_set"]
        else:
            # Fallback for dev mode
            mapping = {"proliferative": 4, "severe": 3, "moderate": 2, "mild": 1, "no_dr": 0}
            c_idx = mapping.get(meta["case_type"], 0)
            pred = class_names[c_idx]
            conf = 0.94
            c_set = [pred]

        cfg = _PRIORITY_CONFIG.get(c_idx, _PRIORITY_CONFIG[0])
        if cfg["priority"] == "P1_CRITICAL":
            p1 += 1
        elif cfg["priority"] == "P2_URGENT":
            p2 += 1
        else:
            p3 += 1

        card = PatientTriageCard(
            patient_id=meta["patient_id"],
            patient_name=meta["patient_name"],
            prediction=pred,
            class_index=c_idx,
            confidence=conf,
            triage_priority=cfg["priority"],
            priority_label=cfg["label"],
            priority_color=cfg["color"],
            referral_urgency=cfg["urgency"],
            recommended_action=cfg["action"],
            conformal_set=c_set,
            icd10_code=cfg["icd10"],
            latency_ms=round((time.perf_counter() - t0) * 1000 / len(cases), 1),
            thumbnail_b64=meta["thumbnail_b64"],
        )
        cards.append(card)

    priority_order = {"P1_CRITICAL": 0, "P2_URGENT": 1, "P3_ROUTINE": 2}
    cards.sort(key=lambda c: (priority_order.get(c.triage_priority, 9), -c.confidence))

    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    return BatchTriageResponse(
        total_patients=len(cards),
        critical_p1_count=p1,
        urgent_p2_count=p2,
        routine_p3_count=p3,
        total_latency_ms=total_ms,
        avg_latency_ms=round(total_ms / len(cards), 1),
        patients=cards,
    )

