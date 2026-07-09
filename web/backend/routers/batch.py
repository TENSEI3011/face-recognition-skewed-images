"""
batch.py — Batch identification router

POST /api/batch/identify  → Upload multiple images OR a ZIP file
                           → Returns results JSON + per-image annotated thumbnails (base64)
GET  /api/batch/export/csv → Download last batch result as CSV
"""

import io
import cv2
import base64
import zipfile
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from typing import List
import csv

from web.backend.services import pipeline_service
from web.backend.services.audit_service import log_event
from web.backend.routers.watchlist import is_watchlisted

router = APIRouter(prefix="/api/batch", tags=["batch"])

# In-memory store for last batch result (for CSV export)
_last_batch: list[dict] = []


def _decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _image_to_b64(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _draw_result(image: np.ndarray, detection, label: str, conf: float,
                 threshold: float) -> np.ndarray:
    vis = image.copy()
    if detection is None:
        return vis
    x, y, w, h = detection["box"]
    x, y = max(0, x), max(0, y)
    is_known = conf >= threshold
    color = (0, 200, 80) if is_known else (0, 60, 220)
    text  = f"{label}  {conf*100:.1f}%"
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(vis, (x, y - th - 10), (x + tw + 8, y), color, -1)
    cv2.putText(vis, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def _process_one(pipe, name: str, data: bytes, threshold: float, top_k: int) -> dict:
    """Run the recognition pipeline on a single image bytes."""
    img = _decode_image(data)
    if img is None:
        return {"filename": name, "error": "Could not decode image.", "detected": False}

    try:
        detection = pipe.detector.detect_largest(img)
        result    = pipe.identify(img, top_k=min(top_k, 5))
    except Exception as e:
        return {"filename": name, "error": str(e), "detected": False}

    if result is None:
        return {
            "filename": name, "detected": False,
            "message":  "No face detected.",
            "thumbnail": _image_to_b64(img),
        }

    labels, confs = result
    top_label = labels[0] if confs and float(confs[0]) >= threshold else "UNKNOWN"
    top_conf  = float(confs[0]) if confs else 0.0
    is_known  = top_conf >= threshold
    watchlist_hit = is_watchlisted(top_label) and is_known

    annotated = _draw_result(img, detection, top_label, top_conf, threshold)

    candidates = [
        {"rank": i + 1, "identity": lbl, "confidence": round(float(c), 4)}
        for i, (lbl, c) in enumerate(zip(labels, confs))
    ]

    # Audit
    log_event("identify", {
        "source":        "batch",
        "filename":      name,
        "identity":      top_label,
        "confidence":    round(top_conf, 4),
        "is_known":      is_known,
        "watchlist_hit": watchlist_hit,
    })

    return {
        "filename":      name,
        "detected":      True,
        "identity":      top_label,
        "confidence":    round(top_conf, 4),
        "is_known":      is_known,
        "watchlist_hit": watchlist_hit,
        "candidates":    candidates,
        "thumbnail":     _image_to_b64(annotated),
    }


@router.post("/identify")
async def batch_identify(
    files:     List[UploadFile] = File(...),
    threshold: float            = Form(0.35),
    top_k:     int              = Form(3),
):
    """
    Identify faces in multiple uploaded images (or a single ZIP file).
    Returns per-image results with annotated thumbnails.
    """
    global _last_batch

    status = pipeline_service.get_status()
    if not status["loaded"]:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")
    pipe = pipeline_service.get_pipeline()

    results = []
    image_items: list[tuple[str, bytes]] = []

    for f in files:
        content = await f.read()
        fname   = f.filename or "image"

        if fname.lower().endswith(".zip"):
            # Extract ZIP and process each image inside
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for member in zf.namelist():
                        ext = Path(member).suffix.lower()
                        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                            image_items.append((Path(member).name, zf.read(member)))
            except Exception:
                results.append({"filename": fname, "error": "Could not open ZIP."})
        else:
            image_items.append((fname, content))

    if not image_items:
        raise HTTPException(status_code=400, detail="No valid images found.")

    # Cap at 50 images per batch
    if len(image_items) > 50:
        image_items = image_items[:50]

    for name, data in image_items:
        res = _process_one(pipe, name, data, threshold, top_k)
        results.append(res)

    _last_batch = results

    known   = sum(1 for r in results if r.get("is_known"))
    unknown = sum(1 for r in results if r.get("detected") and not r.get("is_known"))
    alerts  = sum(1 for r in results if r.get("watchlist_hit"))

    return {
        "results": results,
        "summary": {
            "total":        len(results),
            "detected":     sum(1 for r in results if r.get("detected")),
            "known":        known,
            "unknown":      unknown,
            "watchlist_hits": alerts,
        },
    }


@router.get("/export/csv")
def export_batch_csv():
    """Download the most recent batch results as a CSV file."""
    if not _last_batch:
        raise HTTPException(status_code=404, detail="No batch results yet.")
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["filename", "detected", "identity", "confidence", "is_known", "watchlist_hit", "error"],
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(_last_batch)
    return Response(
        content=buf.getvalue().encode("utf-8"),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_results.csv"},
    )
