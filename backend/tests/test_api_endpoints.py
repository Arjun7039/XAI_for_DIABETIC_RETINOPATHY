"""
Integration tests for FastAPI endpoints: /health, /presets, /predict, and /predict/stream.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_endpoint(async_client: AsyncClient):
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_presets_endpoint_returns_cases(async_client: AsyncClient):
    response = await async_client.get("/presets")
    assert response.status_code == 200
    presets = response.json()
    assert isinstance(presets, list)
    assert len(presets) >= 4
    first = presets[0]
    assert "id" in first
    assert "thumbnail_b64" in first
    assert "stage_label" in first


@pytest.mark.asyncio
async def test_predict_rejects_document_image(async_client: AsyncClient, document_image_bytes: bytes):
    files = {"file": ("document.jpg", document_image_bytes, "image/jpeg")}
    response = await async_client.post("/predict", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["image_quality"] == "poor"
    assert "non_retinal_image" in data["quality_issues"]


@pytest.mark.asyncio
async def test_predict_rejects_blurry_image(async_client: AsyncClient, blurry_image_bytes: bytes):
    files = {"file": ("blurry.jpg", blurry_image_bytes, "image/jpeg")}
    response = await async_client.post("/predict", files=files)
    assert response.status_code == 422
    data = response.json()
    assert data["image_quality"] == "poor"
    assert "blurry" in data["quality_issues"]


@pytest.mark.asyncio
async def test_predict_successful_on_valid_fundus(async_client: AsyncClient, normal_fundus_bytes: bytes):
    files = {"file": ("retina.jpg", normal_fundus_bytes, "image/jpeg")}
    response = await async_client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "gradcam_overlay" in data
    assert "conformal_prediction_set" in data
    assert "agentic_findings" in data
    assert data["agentic_findings"]["consensus_reached"] is True
