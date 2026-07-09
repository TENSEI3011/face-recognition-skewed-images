"""
config_router.py — Pipeline configuration router
GET/POST current pipeline config (modalities, thresholds, etc.)
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from web.backend.services import pipeline_service
from web.backend.config import DEGRADATION_PROFILES

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigUpdate(BaseModel):
    modalities:   Optional[List[str]] = None
    pca_variance: Optional[float]     = None
    svm_kernel:   Optional[str]       = None
    threshold:    Optional[float]     = None
    top_k:        Optional[int]       = None


@router.get("")
def get_config():
    cfg = pipeline_service.get_config()
    return {
        **cfg,
        "available_modalities":   ["hog", "lbp", "geometry", "arcface"],
        "degradation_profiles":   DEGRADATION_PROFILES,
        "available_svm_kernels":  ["rbf", "linear"],
    }


@router.post("")
def update_config(update: ConfigUpdate):
    """Update pipeline configuration and reload the pipeline."""
    changes = {k: v for k, v in update.dict().items() if v is not None}
    if not changes:
        return {"message": "No changes provided.", "config": pipeline_service.get_config()}

    new_cfg = pipeline_service.update_config(changes)
    return {"message": "Configuration updated. Pipeline reloaded.", "config": new_cfg}
