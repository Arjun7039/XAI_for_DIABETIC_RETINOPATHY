"""
Prediction router for RetinaScreen AI.

Endpoint:
  POST /predict — multipart image upload
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.models.gradcam import generate_gradcam_overlay
from app.models.inference import run_inference
from app.models.shap_explainer import generate_shap_overlay
from app.preprocessing.quality_check import check_image_quality
from app.preprocessing.retinal_preprocessing import preprocess_for_inference
from app.schemas.prediction import PredictionResponse, QualityRejectResponse

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse, "description": "Successful DR grading and explainability response"},
        422: {"model": QualityRejectResponse, "description": "Image quality gate failed — retake recommended"},
    },
)
async def predict_retinopathy(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Accepts fundus photograph image file upload, checks image quality,
    runs 5-class DR classification, and generates Grad-CAM + SHAP overlays.
    """
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

    # 1. Quality Gate Check
    passed_quality, quality_issues = check_image_quality(image_bgr)
    if not passed_quality:
        reject_payload = QualityRejectResponse(
            image_quality="poor",
            quality_issues=quality_issues,
            message="Image quality insufficient for reliable grading. Please retake photo.",
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=reject_payload.model_dump(),
        )

    # Retrieve shared model and config instance from app state
    model = getattr(request.app.state, "model", None)
    config = getattr(request.app.state, "config", {})
    class_names = getattr(request.app.state, "class_names", ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"])

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not initialized on the server.",
        )

    # 2. Preprocessing
    use_clahe = config.get("use_clahe", False)
    img_size = config.get("image_size", [224, 224])[0]
    tensor = preprocess_for_inference(image_bgr, use_clahe=use_clahe, img_size=img_size)

    # 3. Model Inference
    (
        prediction,
        class_index,
        confidence,
        probabilities,
        certainty,
        review_recommendation,
    ) = run_inference(model, tensor, class_names=class_names)

    print(f"[DEBUG INFERENCE] Raw Probabilities Dict: {probabilities}")
    print(f"[DEBUG INFERENCE] Selected Class: {prediction} (Index {class_index}) with Confidence: {confidence}")

    # 4. Generate Explainability Overlays (Grad-CAM & SHAP)
    gradcam_b64 = generate_gradcam_overlay(
        model, tensor, image_bgr, target_category=class_index
    )
    shap_b64 = generate_shap_overlay(model, tensor, image_bgr)

    # 5. Construct & Return Response
    response = PredictionResponse(
        prediction=prediction,
        class_index=class_index,
        confidence=confidence,
        probabilities=probabilities,
        certainty=certainty,
        review_recommendation=review_recommendation,
        image_quality="good",
        gradcam_overlay=gradcam_b64,
        shap_overlay=shap_b64,
        model_version=config.get("model_name", "efficientnetv2s-v1"),
    )

    return response
