"""
liveness_router.py
------------------
REST endpoints for the active blink liveness challenge.

Endpoints:
  POST /api/liveness/challenge        — Start a new blink challenge, returns session_id
  POST /api/liveness/verify           — Submit a webcam frame, returns EAR + pass/fail
  GET  /api/liveness/status/{session} — Poll current challenge state
  POST /api/liveness/debug            — Return all 6 passive signal scores for a face crop
"""

import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from web.backend.services.liveness_service import (
    new_blink_challenge,
    get_blink_state,
    process_blink_frame,
    check_liveness,
    check_liveness_detailed,
)
from web.backend import config as _cfg

router = APIRouter(prefix="/api/liveness", tags=["liveness"])


def _decode_frame(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR numpy array."""
    try:
        from PIL import Image, ImageOps
        import io
        pil_img = Image.open(io.BytesIO(data))
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image frame.")
    return img


@router.post("/challenge")
async def start_challenge():
    """
    Start a new blink liveness challenge.

    Returns:
        session_id     : str   — unique ID for this challenge session
        message        : str   — instruction to show the user
        timeout_sec    : float — seconds available to complete
        required_blinks: int   — number of blinks needed
        blink_enabled  : bool  — False if dlib unavailable (passive only)
    """
    session_id = str(uuid.uuid4())
    state = new_blink_challenge(session_id)

    blink_enabled = getattr(_cfg, "BLINK_ENABLED", True)

    return {
        "session_id":      session_id,
        "message":         "👁 Please BLINK naturally to verify you are a real person.",
        "timeout_sec":     state.timeout_sec,
        "required_blinks": state.required_blinks,
        "blink_enabled":   blink_enabled,
    }


@router.post("/verify")
async def verify_blink(
    file:       UploadFile = File(...),
    session_id: str        = Form(...),
):
    """
    Submit a single webcam frame to verify a blink challenge.

    The client should call this endpoint at ~10 fps while showing the
    "Please BLINK" overlay. Stop calling once passed=True or failed=True.

    Returns:
        ear            : float | None   — current Eye Aspect Ratio
        blinks         : int            — blinks detected so far
        passed         : bool           — challenge complete (liveness verified)
        failed         : bool           — timed out
        time_remaining : float          — seconds left
        message        : str            — status message to display
        available      : bool           — False if dlib unavailable
    """
    state = get_blink_state(session_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /api/liveness/challenge first."
        )

    data  = await file.read()
    frame = _decode_frame(data)

    result = process_blink_frame(frame, state)
    return result


@router.get("/status/{session_id}")
async def challenge_status(session_id: str):
    """
    Poll the current state of a blink challenge without submitting a frame.
    Useful for re-checking after a network gap.
    """
    state = get_blink_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id":    session_id,
        "blinks":        state.blinks_detected,
        "passed":        state.passed,
        "failed":        state.failed,
        "time_remaining": round(state.time_remaining, 1),
        "elapsed":       round(state.elapsed, 1),
    }


@router.post("/debug")
async def debug_liveness(file: UploadFile = File(...)):
    """
    Debug endpoint — returns all 6 passive liveness signal scores for an uploaded face.

    Useful for diagnosing why a face is being classified as SPOOF.
    Returns individual scores for: LBP, FFT, Gradient, Color, Specular, Chroma.
    """
    data  = await file.read()
    frame = _decode_frame(data)

    result = check_liveness_detailed(frame)
    return {
        "overall_score": result.get("score"),
        "is_live":       result.get("is_live"),
        "label":         result.get("label"),
        "signals": {
            "lbp_uniformity":   result.get("s_lbp"),
            "fft_frequency":    result.get("s_fft"),
            "gradient_coherence": result.get("s_grad"),
            "color_naturalness":  result.get("s_color"),
            "specular_highlight": result.get("s_spec"),
            "chroma_noise":       result.get("s_chroma"),
        },
        "interpretation": {
            "lbp_uniformity":     "Low = halftone print pattern detected",
            "fft_frequency":      "Low = screen periodic artifacts detected",
            "gradient_coherence": "Low = flat/uniform surface (photo) detected",
            "color_naturalness":  "Low = over-saturated (screen) or clipped (print) colors",
            "specular_highlight": "Low = hard glare hotspot or uniform backlight (screen)",
            "chroma_noise":       "Low = unnaturally smooth chroma (screen pixel grid)",
        }
    }
