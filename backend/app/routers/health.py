"""
Health check router for backend deployment monitoring (Render/Kubernetes).
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(tags=["Health"])


class HealthStatus(BaseModel):
    status: str = "ok"
    service: str = "retinascreen-backend"
    version: str = "1.0.0"


@router.api_route("/health", methods=["GET", "HEAD"], response_model=HealthStatus)
def health_check() -> HealthStatus:
    """Return health status check."""
    return HealthStatus()
