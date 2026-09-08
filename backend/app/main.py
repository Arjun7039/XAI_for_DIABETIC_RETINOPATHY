"""
FastAPI application entry point for RetinaScreen AI.
"""

from __future__ import annotations

import os
import gc
import asyncio

# ── Memory-Lean CPU Thread Allocation for 512MB Cloud Containers (Render Free Tier) ──
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "1")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models.inference import load_model, load_config, warmup_inference
from app.routers import health, predict, presets


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler: load model, free memory, and bind port immediately."""
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

    # Clean up any transient loading allocations to stay well below 512MB RAM
    gc.collect()

    # Pre-compile graph in background task so port 8000 binds instantly on Render
    async def _deferred_warmup():
        await asyncio.sleep(2.0)
        try:
            print("[STARTUP] Pre-compiling graph kernels in background...")
            warmup_inference(model)
            gc.collect()
            print("[STARTUP] High-throughput clinical triage engine online & ready.")
        except Exception as e:
            print(f"[WARN] Deferred warmup notice: {e}")

    asyncio.create_task(_deferred_warmup())

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
