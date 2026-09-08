"""
FastAPI application entry point for RetinaScreen AI.
"""

from __future__ import annotations

import os
import gc

# ── Multi-Core CPU & oneDNN Vector Acceleration for High-Throughput Hospital Triage ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
cpu_cores = str(max(2, min(8, os.cpu_count() or 4)))
os.environ["OMP_NUM_THREADS"] = cpu_cores
os.environ["TF_NUM_INTRAOP_THREADS"] = cpu_cores
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["MALLOC_ARENA_MAX"] = "2"

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.inference import load_model, load_config, warmup_inference
from app.models.gradcam import warmup_gradcam
from app.routers import health, predict, presets


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

    # JIT-compile and warm up inference & Grad-CAM graphs for sub-second hospital triage
    print("[STARTUP] Pre-compiling graph kernels for low-latency clinical triaging...")
    warmup_inference(model)
    warmup_gradcam(model)
    print("[STARTUP] High-throughput clinical triage engine online & ready.")

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
app.include_router(presets.router)


@app.get("/")
def root():
    return {
        "message": "RetinaScreen AI API is running.",
        "docs_url": "/docs",
        "health_check": "/health",
    }
