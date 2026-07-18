"""
config_router.py — Pipeline configuration router
GET/POST current pipeline config (modalities, thresholds, liveness settings, etc.)
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from web.backend.services import pipeline_service
from web.backend.config import DEGRADATION_PROFILES
import web.backend.config as _cfg

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigUpdate(BaseModel):
    modalities:          Optional[List[str]] = None
    pca_variance:        Optional[float]     = None
    svm_kernel:          Optional[str]       = None
    threshold:           Optional[float]     = None
    top_k:               Optional[int]       = None
    liveness_enabled:    Optional[bool]      = None
    liveness_threshold:  Optional[float]     = None


@router.get("")
def get_config():
    cfg = pipeline_service.get_config()
    return {
        **cfg,
        "available_modalities":   ["hog", "lbp", "geometry", "arcface"],
        "degradation_profiles":   DEGRADATION_PROFILES,
        "available_svm_kernels":  ["rbf", "linear"],
        # ── Liveness settings ──────────────────────────────────────────────
        "liveness_enabled":   getattr(_cfg, "LIVENESS_ENABLED",   True),
        "liveness_threshold": getattr(_cfg, "LIVENESS_THRESHOLD", 0.50),
    }


@router.post("")
def update_config(update: ConfigUpdate):
    """Update pipeline configuration and reload the pipeline if needed."""
    changes = {k: v for k, v in update.dict().items() if v is not None}
    if not changes:
        return {"message": "No changes provided.", "config": pipeline_service.get_config()}

    # ── Liveness settings — written directly to config module (in-memory) ──
    # No pipeline retrain needed for these — take effect immediately.
    liveness_changed = False
    if "liveness_enabled" in changes:
        _cfg.LIVENESS_ENABLED = changes.pop("liveness_enabled")
        liveness_changed = True
    if "liveness_threshold" in changes:
        _cfg.LIVENESS_THRESHOLD = changes.pop("liveness_threshold")
        liveness_changed = True

    if liveness_changed:
        # Force liveness_service to reload with new threshold
        try:
            from web.backend.services.liveness_service import reload_detector
            reload_detector()
        except Exception:
            pass

    new_cfg = pipeline_service.update_config(changes) if changes else pipeline_service.get_config()

    return {
        "message": "Configuration updated.",
        "config": {
            **new_cfg,
            "liveness_enabled":   getattr(_cfg, "LIVENESS_ENABLED",   True),
            "liveness_threshold": getattr(_cfg, "LIVENESS_THRESHOLD", 0.50),
        },
    }
