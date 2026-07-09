"""
sr_service.py — Super-Resolution pre-processing service

Uses OpenCV's DNN Super-Resolution (EDSR-x2) to upscale low-resolution UAV faces.
Model (~5 MB) is downloaded on first use to models/sr/.
Falls back gracefully if model or opencv-contrib unavailable.
"""

import urllib.request
from pathlib import Path
import numpy as np

# ── Model config ──────────────────────────────────────────────────────────────
_MODEL_URL  = "https://github.com/nicehash/EDSR_OpenCV/raw/main/EDSR_x2.pb"
_ROOT       = Path(__file__).resolve().parents[3]      # project root
_MODEL_DIR  = _ROOT / "models" / "sr"
_MODEL_PATH = _MODEL_DIR / "EDSR_x2.pb"

_sr_instance = None   # lazily loaded singleton


def _load_sr():
    """Download model if missing, then load OpenCV DNN Super-Resolution."""
    global _sr_instance
    if _sr_instance is not None:
        return _sr_instance

    try:
        from cv2 import dnn_superres
    except ImportError:
        return None   # opencv-contrib not installed

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not _MODEL_PATH.exists():
        try:
            print(f"[SR] Downloading EDSR x2 model to {_MODEL_PATH} …")
            urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
            print("[SR] Model downloaded successfully.")
        except Exception as e:
            print(f"[SR] Download failed: {e}")
            return None

    try:
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(str(_MODEL_PATH))
        sr.setModel("edsr", 2)
        _sr_instance = sr
        print("[SR] Super-Resolution model loaded (EDSR x2).")
        return sr
    except Exception as e:
        print(f"[SR] Failed to load model: {e}")
        return None


def upscale(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Upscale an image using EDSR-x2 super-resolution.
    Returns the upscaled image, or the original if SR is unavailable.
    """
    sr = _load_sr()
    if sr is None:
        return image
    try:
        return sr.upsample(image)
    except Exception:
        return image


def is_available() -> bool:
    """Return True if the SR model is loaded and ready."""
    return _load_sr() is not None
