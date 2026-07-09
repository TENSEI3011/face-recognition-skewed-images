"""
identify.py — Face identification router
Handles: single image upload → identity prediction with bounding box overlay

Enhanced with:
  - Audit logging (every identification is recorded)
  - Watchlist hit detection
  - Super-Resolution toggle (use_sr parameter)
"""

import cv2
import base64
import numpy as np
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from web.backend.services import pipeline_service
from web.backend.services.audit_service import log_event
from web.backend.routers.watchlist import is_watchlisted
from web.backend.config import DEFAULT_TOP_K, DEGRADATION_PROFILES

router = APIRouter(prefix="/api/identify", tags=["identify"])


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def _apply_degradation(image: np.ndarray, profile: str) -> np.ndarray:
    """Apply UAV degradation simulation if a profile is selected."""
    if profile == "CLEAN" or profile not in DEGRADATION_PROFILES:
        return image
    try:
        from src.augmentation import UAVAugmentor
        aug = UAVAugmentor(profile=profile)
        return aug.augment(image)
    except Exception:
        return image


def _apply_sr(image: np.ndarray, use_sr: bool) -> np.ndarray:
    """Optionally apply Super-Resolution upscaling."""
    if not use_sr:
        return image
    try:
        from web.backend.services.sr_service import upscale
        return upscale(image)
    except Exception:
        return image


def _draw_result(image: np.ndarray, detection, labels, confs, threshold: float) -> np.ndarray:
    """Draw bounding box and identity label on the image."""
    vis = image.copy()
    if detection is None:
        return vis

    x, y, w, h = detection["box"]
    x, y = max(0, x), max(0, y)

    top_label = labels[0] if labels else "UNKNOWN"
    top_conf  = confs[0]  if confs  else 0.0
    is_known  = top_conf >= threshold

    color = (0, 200, 80) if is_known else (0, 60, 220)
    label_text = f"{top_label}  {top_conf*100:.1f}%" if is_known else f"UNKNOWN  {top_conf*100:.1f}%"

    # Draw box
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

    # Background for label
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(vis, (x, y - th - 10), (x + tw + 8, y), color, -1)
    cv2.putText(vis, label_text, (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis


def _image_to_b64(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@router.post("")
async def identify_face(
    file:        UploadFile = File(...),
    top_k:       int        = Form(DEFAULT_TOP_K),
    threshold:   float      = Form(0.35),
    degradation: str        = Form("CLEAN"),
    use_sr:      bool       = Form(False),
):
    """
    Identify the face in the uploaded image.
    Returns: top-k predictions, confidence scores, annotated image (base64).
    Also logs the event to audit_log.jsonl and checks the watchlist.
    """
    status = pipeline_service.get_status()
    if not status["loaded"]:
        raise HTTPException(
            status_code=503,
            detail=status.get("error", "Pipeline not loaded. Run baseline experiment first.")
        )

    pipe = pipeline_service.get_pipeline()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline unavailable.")

    # Decode uploaded image
    data  = await file.read()
    image = _decode_image(data)

    # Apply Super-Resolution (before degradation for best quality)
    image = _apply_sr(image, use_sr)

    # Apply UAV degradation
    image_deg = _apply_degradation(image, degradation)

    # Run detection for bounding box overlay
    detection = pipe.detector.detect_largest(image_deg)

    # Run full identification
    result = pipe.identify(image_deg, top_k=min(top_k, 5))

    if result is None:
        log_event("identify", {
            "source":     "identify",
            "filename":   file.filename or "upload",
            "detected":   False,
            "degradation": degradation,
            "use_sr":     use_sr,
        })
        return {
            "detected": False,
            "message":  "No face detected in the image.",
            "annotated_image": _image_to_b64(image_deg),
            "candidates": [],
        }

    labels, confs = result

    # Draw annotations
    annotated = _draw_result(image_deg, detection, labels, confs, threshold)

    candidates = [
        {"rank": i + 1, "identity": lbl, "confidence": round(float(conf), 4)}
        for i, (lbl, conf) in enumerate(zip(labels, confs))
    ]

    top_conf  = float(confs[0]) if confs else 0.0
    top_label = labels[0] if top_conf >= threshold else "UNKNOWN"
    is_known  = top_conf >= threshold

    # Watchlist check
    watchlist_hit = is_watchlisted(top_label) and is_known

    # Audit log
    log_event("identify", {
        "source":        "identify",
        "filename":      file.filename or "upload",
        "identity":      top_label,
        "confidence":    round(top_conf, 4),
        "is_known":      is_known,
        "watchlist_hit": watchlist_hit,
        "degradation":   degradation,
        "use_sr":        use_sr,
    })

    return {
        "detected":        True,
        "identity":        top_label,
        "confidence":      round(top_conf, 4),
        "is_known":        is_known,
        "watchlist_hit":   watchlist_hit,
        "threshold":       threshold,
        "candidates":      candidates,
        "annotated_image": _image_to_b64(annotated),
        "degradation":     degradation,
        "use_sr":          use_sr,
    }


@router.post("/frame")
async def identify_frame(
    file:      UploadFile = File(...),
    threshold: float      = Form(0.35),
):
    """
    Fast single-frame identification endpoint for WebSocket-like HTTP polling.
    Returns minimal JSON (no annotated image — caller draws on canvas).
    """
    status = pipeline_service.get_status()
    if not status["loaded"]:
        return {"detected": False, "error": "Pipeline not loaded."}

    pipe = pipeline_service.get_pipeline()
    if pipe is None:
        return {"detected": False, "error": "Pipeline unavailable."}

    data  = await file.read()
    image = _decode_image(data)

    # Detect all faces for live mode
    detections = pipe.detector.detect(image)
    if not detections:
        return {"detected": False, "faces": []}

    faces_result = []
    for det in detections[:4]:  # Process max 4 faces per frame
        try:
            aligned = pipe.aligner.align_from_detection(image, det)
            if aligned is None:
                continue
            labels, confs = pipe.identify_from_aligned(aligned, top_k=1)
            top_conf  = float(confs[0]) if confs else 0.0
            top_label = labels[0] if confs and top_conf >= threshold else "UNKNOWN"
            x, y, w, h = det["box"]
            faces_result.append({
                "box":        {"x": x, "y": y, "w": w, "h": h},
                "identity":   top_label,
                "confidence": round(top_conf, 4),
                "is_known":   top_conf >= threshold,
            })
        except Exception:
            pass

    return {"detected": len(faces_result) > 0, "faces": faces_result}
