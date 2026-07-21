"""
liveness_service.py
-------------------
Singleton wrapper around PassiveLivenessDetector and BlinkDetector for use
by the web backend.

Provides:
  check_liveness(face_img)         — passive 6-signal anti-spoofing check
  new_blink_challenge(session_id)  — start an active blink challenge session
  process_blink_frame(frame, state)— process one webcam frame against challenge
  get_blink_state(session_id)      — retrieve existing challenge state

Usage:
    from web.backend.services.liveness_service import check_liveness, new_blink_challenge

    # Passive check (always runs)
    result = check_liveness(face_crop_bgr)
    # → {"score": 0.72, "is_live": True, "label": "REAL"}

    # Active blink challenge (webcam mode)
    state = new_blink_challenge("sess-abc123")
    # ... per frame:
    result = process_blink_frame(frame_bgr, state)
    # → {"ear": 0.28, "blinks": 1, "passed": True, ...}
"""

import numpy as np
import time
from typing import Optional

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.liveness import PassiveLivenessDetector
from src.blink_detector import BlinkDetector, BlinkChallengeState
from web.backend import config as _cfg

# ── Passive liveness singleton ────────────────────────────────────────────────
_detector: Optional[PassiveLivenessDetector] = None


def _get_detector() -> PassiveLivenessDetector:
    """Lazy-load the passive liveness detector (created only once)."""
    global _detector
    if _detector is None:
        threshold = getattr(_cfg, "LIVENESS_THRESHOLD", 0.45)
        _detector = PassiveLivenessDetector(threshold=threshold)
        print(f"[LivenessService] Passive liveness detector initialised "
              f"(threshold={threshold}, 6-signal fusion)")
    return _detector


def reload_detector() -> None:
    """Force re-creation of the passive detector (e.g., after config change)."""
    global _detector
    _detector = None
    _get_detector()


# ── Blink detector singleton ──────────────────────────────────────────────────
_blink_detector: Optional[BlinkDetector] = None
_blink_sessions: dict = {}   # session_id → BlinkChallengeState


def _get_blink_detector() -> BlinkDetector:
    """Lazy-load the blink detector (created only once)."""
    global _blink_detector
    if _blink_detector is None:
        predictor_path = str(getattr(_cfg, "PREDICTOR_PATH", ROOT / "models" / "shape_predictor_68_face_landmarks.dat"))
        ear_threshold  = getattr(_cfg, "BLINK_EAR_THRESHOLD",  0.25)
        timeout_sec    = getattr(_cfg, "BLINK_TIMEOUT_SEC",     7.0)
        required       = getattr(_cfg, "BLINK_REQUIRED_COUNT",  1)
        consec         = getattr(_cfg, "BLINK_CONSEC_FRAMES",   2)

        _blink_detector = BlinkDetector(
            predictor_path  = predictor_path,
            ear_threshold   = ear_threshold,
            timeout_sec     = timeout_sec,
            required_blinks = required,
            consec_frames   = consec,
        )
        status = "OK" if _blink_detector.is_available else "UNAVAILABLE (dlib error)"
        print(f"[LivenessService] BlinkDetector initialised — status: {status}")
    return _blink_detector


# ── Passive liveness public API ───────────────────────────────────────────────

def check_liveness(face_img: np.ndarray) -> dict:
    """
    Run passive liveness check on a face crop.

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


def check_liveness_detailed(face_img: np.ndarray) -> dict:
    """
    Extended passive liveness check that returns all 6 individual signal scores.
    Useful for debugging why a face is being rejected.
    """
    if not getattr(_cfg, "LIVENESS_ENABLED", True):
        return {"score": 1.0, "is_live": True, "label": "REAL (disabled)",
                "s_lbp": 1.0, "s_fft": 1.0, "s_grad": 1.0,
                "s_color": 1.0, "s_spec": 1.0, "s_chroma": 1.0}

    if face_img is None or face_img.size == 0:
        return {"score": 0.0, "is_live": False, "label": "SPOOF",
                "s_lbp": 0.0, "s_fft": 0.0, "s_grad": 0.0,
                "s_color": 0.0, "s_spec": 0.0, "s_chroma": 0.0}

    detector = _get_detector()
    result = detector.predict_detailed(face_img)
    result["is_live"] = result["label"] == "REAL"
    return result


# ── Active blink challenge public API ─────────────────────────────────────────

def new_blink_challenge(session_id: str) -> BlinkChallengeState:
    """
    Create a new blink challenge for a webcam session.

    Parameters
    ----------
    session_id : str
        Unique identifier for this webcam session (e.g. UUID from frontend).

    Returns
    -------
    BlinkChallengeState — the newly created state object.
    """
    # Clean up old sessions (older than 5 minutes)
    now = time.time()
    expired = [sid for sid, s in _blink_sessions.items()
               if now - s.started_at > 300]
    for sid in expired:
        del _blink_sessions[sid]

    detector = _get_blink_detector()
    state = detector.new_challenge(session_id)
    _blink_sessions[session_id] = state
    return state


def get_blink_state(session_id: str) -> Optional[BlinkChallengeState]:
    """Retrieve an existing blink challenge state by session ID."""
    return _blink_sessions.get(session_id)


def process_blink_frame(frame_bgr: np.ndarray, state: BlinkChallengeState) -> dict:
    """
    Process a single webcam frame against an active blink challenge.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Full webcam frame (BGR), not just the face crop.
        BlinkDetector runs its own face detection internally.

    state : BlinkChallengeState
        The session state returned by new_blink_challenge().

    Returns
    -------
    dict:
        ear            : float | None   — current EAR value
        blinks         : int            — blinks detected so far
        passed         : bool           — challenge complete
        failed         : bool           — timed out
        time_remaining : float          — seconds left
        message        : str            — human-readable status
        available      : bool           — False if dlib unavailable (fallback to passive)
    """
    if not getattr(_cfg, "BLINK_ENABLED", True):
        # Blink challenge disabled — auto-pass
        state.passed = True
        return {
            "ear": None, "blinks": 0, "passed": True, "failed": False,
            "time_remaining": 0.0, "message": "Blink check disabled",
            "available": False,
        }

    detector = _get_blink_detector()

    if not detector.is_available:
        # dlib not available — gracefully auto-pass (passive check is still active)
        state.passed = True
        return {
            "ear": None, "blinks": 0, "passed": True, "failed": False,
            "time_remaining": 0.0,
            "message": "Blink detector unavailable — passive check only",
            "available": False,
        }

    result = detector.process_frame(frame_bgr, state)
    result["available"] = True
    return result
