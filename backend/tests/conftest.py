"""
Pytest configuration and test fixtures for RetinaScreen AI.
"""

import os
import sys

# Ensure local backend and tests directories take precedence over site-packages
tests_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(tests_dir)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from helpers import create_synthetic_fundus, to_jpeg_bytes


@pytest.fixture
def normal_fundus_bytes() -> bytes:
    return to_jpeg_bytes(create_synthetic_fundus("normal"))


@pytest.fixture
def blurry_image_bytes() -> bytes:
    return to_jpeg_bytes(create_synthetic_fundus("blur"))


@pytest.fixture
def document_image_bytes() -> bytes:
    return to_jpeg_bytes(create_synthetic_fundus("document"))


@pytest.fixture
def lesions_fundus_bytes() -> bytes:
    return to_jpeg_bytes(create_synthetic_fundus("lesions"))


@pytest.fixture
async def async_client():
    # Ensure app state has fallback model for testing
    if getattr(app.state, "model", None) is None:
        from app.models.inference import load_config, load_model
        cfg = load_config("weights/efficientnet_b4_config.json")
        app.state.config = cfg
        app.state.model = load_model(cfg)
        app.state.class_names = cfg.get(
            "class_names",
            ["Mild_NPDR", "Moderate_NPDR", "No_DR", "Proliferative_DR", "Severe_NPDR"],
        )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
