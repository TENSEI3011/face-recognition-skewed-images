"""
identify.py — Face identification router
Handles: single image upload → identity prediction with bounding box overlay

Enhanced with:
  - Audit logging (every identification is recorded)
  - Watchlist hit detection
  - Super-Resolution toggle (use_sr parameter)
  - FAISS cosine matcher (primary identification gate — replaces SVM open-set gap)
  - Raised COSINE_THRESHOLD 0.30 → 0.60 (safety-critical: reduces false positives)
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
from web.backend.config import DEFAULT_TOP_K, DEGRADATION_PROFILES, FAISS_THRESHOLD

router = APIRouter(prefix="/api/identify", tags=["identify"])


def _decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array with EXIF rotation correction.

    Phone cameras (iOS/Android) store a rotation tag in EXIF metadata.
    OpenCV/cv2.imdecode ignores this tag, so selfies appear rotated.
    Pillow reads the EXIF tag and applies the correct rotation first.
    """
    # Try Pillow first for EXIF-aware decoding
    try:
        from PIL import Image, ImageOps
        import io
        pil_img = Image.open(io.BytesIO(data))
        # Apply EXIF orientation (rotates to correct upright orientation)
        pil_img = ImageOps.exif_transpose(pil_img)
        # Convert to RGB then BGR for OpenCV
        pil_img = pil_img.convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if img is not None and img.size > 0:
            return img
    except Exception:
        pass  # Fall back to raw OpenCV decode

    # Fallback: plain OpenCV decode (no EXIF rotation)
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


def _apply_enhance(image: np.ndarray, use_enhance: bool, enhance_mode: str = "auto") -> np.ndarray:
    """Optionally apply deblur + dehaze + CLAHE enhancement before detection."""
    if not use_enhance:
        return image
    try:
        from web.backend.services.enhance_service import enhance
        return enhance(image, mode=enhance_mode)
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
    file:         UploadFile = File(...),
    top_k:        int        = Form(DEFAULT_TOP_K),
    threshold:    float      = Form(0.65),
    degradation:  str        = Form("CLEAN"),
    use_sr:       bool       = Form(False),
    use_enhance:  bool       = Form(False),
    enhance_mode: str        = Form("auto"),
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

    # Apply Enhancement (deblur + dehaze + CLAHE) — before SR for best results
    image = _apply_enhance(image, use_enhance, enhance_mode)

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
            "source":       "identify",
            "filename":     file.filename or "upload",
            "detected":     False,
            "degradation":  degradation,
            "use_sr":       use_sr,
            "use_enhance":  use_enhance,
            "enhance_mode": enhance_mode,
        })
        return {
            "detected": False,
            "message":  "No face detected in the image.",
            "annotated_image": _image_to_b64(image_deg),
            "candidates": [],
            "use_enhance":  use_enhance,
            "enhance_mode": enhance_mode,
        }

    labels, confs = result

    svm_conf  = float(confs[0]) if confs else 0.0
    svm_label = labels[0] if labels else "UNKNOWN"

    # Cosine similarity rejection using in-memory gallery embeddings.
    # These were stored at retrain time using the SAME ArcFace extraction
    # as inference, so they are directly comparable.
    is_known  = False
    top_label = "UNKNOWN"
    top_conf  = svm_conf
    q_feat    = None   # ArcFace embedding — computed once, reused for top-k candidates

    # ── Threshold: use FAISS_THRESHOLD from config (default 0.40 for outdoor/UAV) ─
    # 0.60 was too strict for real-world balcony/outdoor video conditions where
    # lighting, angle and compression reduce cosine similarity vs gallery photos.
    # config.FAISS_THRESHOLD = 0.40 balances FAR vs FRR for these conditions.
    COSINE_THRESHOLD = FAISS_THRESHOLD

    # ── Path 1: FAISS matcher (preferred — handles open-set correctly) ────────
    faiss_matcher = pipeline_service.get_faiss_matcher()
    if faiss_matcher is not None and faiss_matcher.is_built:
        try:
            import numpy as _np
            detection_inner = pipe.detector.detect_largest(image_deg)
            if detection_inner is not None:
                aligned_q = pipe.aligner.align_from_detection(image_deg, detection_inner)
                if aligned_q is not None and pipe.arc_ext:
                    q_feat = pipe.arc_ext.extract(aligned_q)
                    faiss_label, faiss_sim = faiss_matcher.best_match(q_feat)
                    # faiss_matcher.best_match() applies FAISS_THRESHOLD internally
                    # "UNKNOWN" means similarity was below threshold
                    is_known  = (faiss_label != "UNKNOWN")
                    top_label = faiss_label
                    top_conf  = round(faiss_sim, 4)
        except Exception as _e:
            print(f"[Identify] FAISS matching failed: {_e} — falling back to cosine/SVM")
            faiss_matcher = None   # trigger fallback below


    # ── Path 2: Manual cosine gate over stored gallery embeddings (fallback) ──
    if faiss_matcher is None or not faiss_matcher.is_built:
        gallery_arc = pipeline_service.get_gallery_arc_features()
        if gallery_arc:
            try:
                import numpy as _np
                detection_inner = pipe.detector.detect_largest(image_deg)
                if detection_inner is not None:
                    aligned_q = pipe.aligner.align_from_detection(image_deg, detection_inner)
                    if aligned_q is not None and pipe.arc_ext:
                        q_feat = pipe.arc_ext.extract(aligned_q)
                        q_norm = _np.linalg.norm(q_feat)
                        if q_norm > 1e-6:
                            q_feat = q_feat / q_norm
                            best_sim  = -1.0
                            best_iden = "UNKNOWN"
                            for identity, feats in gallery_arc.items():
                                mat  = _np.array(feats)
                                sims = mat @ q_feat
                                sim  = float(_np.max(sims))
                                if sim > best_sim:
                                    best_sim  = sim
                                    best_iden = identity
                            if best_sim >= COSINE_THRESHOLD:
                                is_known  = True
                                top_label = best_iden
                                top_conf  = round(best_sim, 4)
            except Exception:
                is_known  = svm_conf >= threshold
                top_label = svm_label if is_known else "UNKNOWN"
                top_conf  = svm_conf
        else:
            # Gallery embeddings not in memory yet — use SVM score only
            is_known  = svm_conf >= threshold
            top_label = svm_label if is_known else "UNKNOWN"
            top_conf  = svm_conf

    # ── Build ranked candidates list ────────────────────────────────────────────
    # IMPORTANT: Use FAISS top-k when FAISS is the primary matcher so that the
    # ranked list agrees with the main identified label (which also comes from FAISS).
    # Previously the candidates list used SVM output, which often disagreed with
    # the FAISS-identified label shown in the header (e.g. header=Siddhant but
    # rank#1=Aditi). Now both use the same source.
    candidates = []
    if faiss_matcher is not None and faiss_matcher.is_built and q_feat is not None:
        try:
            # Use search_ranked() so ranked list always shows real identity names
            # and scores (not "UNKNOWN") — this makes Rank #1 agree with the
            # primary identified label shown in the header.
            faiss_results = faiss_matcher.search_ranked(q_feat, top_k=min(top_k, 5))
            for i, r in enumerate(faiss_results):
                candidates.append({
                    "rank":       i + 1,
                    "identity":   r["identity"],
                    "confidence": r["similarity"],
                    "is_known":   r["is_known"],
                })
        except Exception:
            # Fallback to SVM if FAISS top-k fails
            candidates = [
                {"rank": i + 1, "identity": lbl, "confidence": round(float(conf), 4)}
                for i, (lbl, conf) in enumerate(zip(labels, confs))
            ]
    else:
        # No FAISS — use SVM ranked output
        candidates = [
            {"rank": i + 1, "identity": lbl, "confidence": round(float(conf), 4)}
            for i, (lbl, conf) in enumerate(zip(labels, confs))
        ]

    # Draw bounding box and label on the image
    annotated = _draw_result(image_deg, detection, [top_label], [top_conf], threshold)



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
        "use_enhance":   use_enhance,
        "enhance_mode":  enhance_mode,
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
        "use_enhance":     use_enhance,
        "enhance_mode":    enhance_mode,
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
