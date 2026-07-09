"""
pipeline_service.py
-------------------
Singleton wrapper around FaceRecognitionPipeline.

Loads the trained pipeline ONCE on startup and provides thread-safe
access for all API endpoints.  Keeps the heavy model objects (ArcFace,
dlib) in memory between requests so inference is instant.
"""

import sys
import json
import threading
from pathlib import Path
from typing import Optional

# Add project root to path so `src` imports work
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from web.backend.config import (
    MODELS_DIR, PREDICTOR_PATH,
    DEFAULT_MODALITIES, DEFAULT_PCA_VARIANCE, DEFAULT_SVM_KERNEL,
)

from src.pipeline import FaceRecognitionPipeline

# ── Module-level singleton state ───────────────────────────────────────────────
_pipeline: Optional[FaceRecognitionPipeline] = None
_lock = threading.Lock()

# Current config (can be changed via /api/config)
_current_config = {
    "modalities":    DEFAULT_MODALITIES.copy(),
    "pca_variance":  DEFAULT_PCA_VARIANCE,
    "svm_kernel":    DEFAULT_SVM_KERNEL,
    "threshold":     0.35,
    "top_k":         3,
}

_status = {
    "loaded":   False,
    "error":    None,
    "models_dir": str(MODELS_DIR),
    "modalities": DEFAULT_MODALITIES.copy(),
}


def get_pipeline() -> Optional[FaceRecognitionPipeline]:
    """Return the loaded pipeline, or None if not yet loaded."""
    return _pipeline


def get_status() -> dict:
    return _status.copy()


def get_config() -> dict:
    return _current_config.copy()


def update_config(new_config: dict) -> dict:
    """Merge new_config into current config and reload pipeline."""
    global _current_config
    _current_config.update(new_config)
    # Reload with new settings
    _reload_pipeline()
    return _current_config.copy()


def load_pipeline() -> None:
    """
    Load the trained pipeline from disk.  Called once at startup.
    Thread-safe — uses a lock to prevent double-loading.
    """
    global _pipeline, _status
    with _lock:
        if _pipeline is not None:
            return  # Already loaded
        _do_load()


def _reload_pipeline() -> None:
    """Force-reload pipeline with current config (called after config update)."""
    global _pipeline
    with _lock:
        _pipeline = None
        _do_load()


def _do_load() -> None:
    """Internal load — must be called with _lock held."""
    global _pipeline, _status

    models_dir = Path(_current_config.get("models_dir", str(MODELS_DIR)))

    try:
        # Read saved modalities from disk if available
        meta_path = models_dir / "modalities.json"
        if meta_path.exists():
            with open(meta_path) as f:
                saved_modalities = json.load(f).get("modalities", DEFAULT_MODALITIES)
        else:
            saved_modalities = _current_config["modalities"]

        pipe = FaceRecognitionPipeline(
            modalities=saved_modalities,
            pca_variance=_current_config["pca_variance"],
            svm_kernel=_current_config["svm_kernel"],
            predictor_path=str(PREDICTOR_PATH),
        )

        # Check if trained models exist
        if (models_dir / "svm_classifier.pkl").exists():
            pipe.load(models_dir)
            _status["loaded"]     = True
            _status["error"]      = None
            _status["modalities"] = saved_modalities
        else:
            _status["loaded"] = False
            _status["error"]  = f"No trained model found at {models_dir}. Run a baseline experiment first."

        _pipeline = pipe

    except Exception as e:
        _status["loaded"] = False
        _status["error"]  = str(e)
        _pipeline = None
