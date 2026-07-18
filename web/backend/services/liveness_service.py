"""
liveness_service.py
-------------------
Singleton wrapper around PassiveLivenessDetector for use by the web backend.

Provides a single shared detector instance across all routers so the
PassiveLivenessDetector is only initialised once at startup.

Usage:
    from web.backend.services.liveness_service import check_liveness

    result = check_liveness(face_crop_bgr)
    # result = {
    #     "score":   0.72,        # 0.0 (spoof) → 1.0 (real)
    #     "is_live": True,
    #     "label":   "REAL",      # or "SPOOF"
    # }
"""

import numpy as np
from typing import Optional

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.liveness import PassiveLivenessDetector
from web.backend import config as _cfg

# ── Singleton ─────────────────────────────────────────────────────────────────
_detector: Optional[PassiveLivenessDetector] = None


def _get_detector() -> PassiveLivenessDetector:
    """Lazy-load the liveness detector (created only once)."""
    global _detector
    if _detector is None:
        threshold = getattr(_cfg, "LIVENESS_THRESHOLD", 0.50)
        _detector = PassiveLivenessDetector(threshold=threshold)
        print(f"[LivenessService] Passive liveness detector initialised "
              f"(threshold={threshold})")
    return _detector


def reload_detector() -> None:
    """Force re-creation of the detector (e.g., after config change)."""
    global _detector
    _detector = None
    _get_detector()


# ── Public API ────────────────────────────────────────────────────────────────

def check_liveness(face_img: np.ndarray) -> dict:
    """
    Run liveness check on a face crop.

    Parameters
    ----------
    face_img : np.ndarray
        BGR (or grayscale) face image — any size, typically 112×112 aligned crop.

    Returns
    -------
    dict with keys:
        score   : float  — 0.0 (certain spoof) to 1.0 (certain real)
        is_live : bool   — True if score >= threshold
        label   : str    — "REAL" or "SPOOF"
    """
    # If liveness is globally disabled, always return REAL (pass-through)
    if not getattr(_cfg, "LIVENESS_ENABLED", True):
        return {"score": 1.0, "is_live": True, "label": "REAL (disabled)"}

    if face_img is None or face_img.size == 0:
        return {"score": 0.0, "is_live": False, "label": "SPOOF"}

    detector = _get_detector()
    score, label = detector.predict(face_img)
    is_live = label == "REAL"

    return {
        "score":   round(float(score), 4),
        "is_live": is_live,
        "label":   label,
    }
