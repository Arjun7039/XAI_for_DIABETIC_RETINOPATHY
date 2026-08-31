"""
FastAPI application entry point for RetinaScreen AI.
"""

from __future__ import annotations

import os
import gc

# ── Memory & CPU Optimization for 512MB RAM Containers (Render Free Tier) ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.inference import load_model, load_config
from app.routers import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: load TensorFlow model and config on startup."""
    # Load Config first
    config_path = os.getenv(
        "MODEL_CONFIG_PATH", "backend/weights/efficientnet_b4_config.json"
    )
    if not os.path.exists(config_path) and os.path.exists("weights/efficientnet_b4_config.json"):
        config_path = "weights/efficientnet_b4_config.json"
        
    config = load_config(config_path)

    print(f"[STARTUP] Initializing models based on config: {config_path}")
    model = load_model(config)
    class_names = config.get("class_names", ["Mild_NPDR", "Moderate_NPDR", "No_DR", "Proliferative_DR", "Severe_NPDR"])

    app.state.model = model
    app.state.config = config
    app.state.class_names = class_names

    yield

    print("[SHUTDOWN] Cleaning up server resources.")


app = FastAPI(
    title="RetinaScreen AI Backend",
    description="Explainable, uncertainty-aware Diabetic Retinopathy screening API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in origins_raw.split(",")] if origins_raw != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/")
def root():
    return {
        "message": "RetinaScreen AI API is running.",
        "docs_url": "/docs",
        "health_check": "/health",
    }
