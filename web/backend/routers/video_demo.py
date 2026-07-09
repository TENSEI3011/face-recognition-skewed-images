"""
video_demo.py — Video processing + WebSocket live stream router

Three modes:
  1. POST /api/demo/process    — Upload video → process → return annotated video URL
  2. WebSocket /ws/stream      — Real-time webcam frame analysis
  3. GET  /api/demo/output/{filename} — Download processed video
"""

import cv2
import json
import base64
import asyncio
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from web.backend.services import pipeline_service
from web.backend.config import DEFAULT_TOP_K

router = APIRouter(tags=["demo"])

# Temp dir for processed videos
OUTPUT_DIR = ROOT / "web" / "backend" / "_video_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _draw_faces(frame: np.ndarray, faces: list, threshold: float) -> np.ndarray:
    vis = frame.copy()
    for face in faces:
        box    = face.get("box", {})
        x, y, w, h = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
        label  = face.get("identity", "UNKNOWN")
        conf   = face.get("confidence", 0.0)
        known  = face.get("is_known", False)
        color  = (0, 200, 80) if known else (0, 60, 220)
        text   = f"{label}  {conf*100:.1f}%"

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (x, y - th - 10), (x + tw + 8, y), color, -1)
        cv2.putText(vis, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return vis


def _frame_to_b64(frame: np.ndarray, quality: int = 75) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _predict_faces(pipe, frame: np.ndarray, threshold: float) -> list:
    """Run detection + recognition on a downscaled copy, return boxes in original coords.

    Recognition strategy:
      1. MongoDB connected → ArcFace embedding + cosine similarity (no SVM needed)
      2. MongoDB not connected → SVM pipeline (original behaviour)
    """
    from web.backend.services.mongo_service import is_connected
    use_mongo = is_connected()

    orig_h, orig_w = frame.shape[:2]

    # ── Downscale to speed up inference ──────────────────────────────────────
    MAX_SIDE = 320
    scale    = min(MAX_SIDE / orig_w, MAX_SIDE / orig_h, 1.0)  # never upscale
    if scale < 1.0:
        small = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)),
                           interpolation=cv2.INTER_LINEAR)
    else:
        small = frame
        scale = 1.0

    detections = pipe.detector.detect(small)
    results = []
    for det in detections[:4]:
        try:
            aligned = pipe.aligner.align_from_detection(small, det)
            if aligned is None:
                continue

            if use_mongo and pipe.arc_ext:
                # ── MongoDB path: cosine similarity ───────────────────────
                from web.backend.services.embedding_service import find_best_match
                emb    = pipe.arc_ext.extract(aligned)
                match  = find_best_match(emb, threshold=threshold)
                top_label = match["identity"]
                top_conf  = match["confidence"]
            else:
                # ── SVM fallback ──────────────────────────────────────────
                labels, confs = pipe.identify_from_aligned(aligned, top_k=1)
                top_conf  = float(confs[0]) if confs else 0.0
                top_label = labels[0] if confs and top_conf >= threshold else "UNKNOWN"

            # Scale box back to original frame coordinates
            bx, by, bw, bh = det["box"]
            x = int(bx / scale)
            y = int(by / scale)
            w = int(bw / scale)
            h = int(bh / scale)

            results.append({
                "box":        {"x": max(0, x), "y": max(0, y), "w": w, "h": h},
                "identity":   top_label,
                "confidence": round(top_conf, 4),
                "is_known":   top_conf >= threshold,
            })
        except Exception:
            pass
    return results



# ── WebSocket live stream ──────────────────────────────────────────────────────

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time face recognition stream.
    Client sends JPEG frames as binary; server responds with JSON containing
    face bounding boxes + identities, and an annotated JPEG frame.
    """
    await websocket.accept()

    status = pipeline_service.get_status()
    if not status["loaded"]:
        await websocket.send_json({"error": "Pipeline not loaded.", "faces": []})
        await websocket.close()
        return

    pipe = pipeline_service.get_pipeline()
    threshold = 0.35

    try:
        while True:
            # Receive raw frame bytes from browser
            data = await websocket.receive_bytes()

            # Check for control messages (JSON config updates)
            try:
                msg = json.loads(data.decode("utf-8"))
                if "threshold" in msg:
                    threshold = float(msg["threshold"])
                    await websocket.send_json({"type": "config_ack", "threshold": threshold})
                continue
            except Exception:
                pass  # Not JSON, treat as image bytes

            # Decode frame
            arr   = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Run inference in thread pool to avoid blocking event loop
            loop  = asyncio.get_running_loop()
            faces = await loop.run_in_executor(None, _predict_faces, pipe, frame, threshold)

            # Draw on frame at ORIGINAL resolution (boxes already scaled back)
            annotated = _draw_faces(frame, faces, threshold)

            response = {
                "type":   "frame",
                "faces":  faces,
                "frame":  _frame_to_b64(annotated, quality=75),
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e), "faces": []})
        except Exception:
            pass


# ── Video file processing ──────────────────────────────────────────────────────

@router.post("/api/demo/process")
async def process_video(
    file:       UploadFile = File(...),
    threshold:  float      = Form(0.35),
    process_every: int     = Form(3),
):
    """
    Upload a video → run face recognition on every Nth frame → return annotated video.
    """
    status = pipeline_service.get_status()
    if not status["loaded"]:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    pipe = pipeline_service.get_pipeline()

    # Save uploaded file to temp location
    suffix  = Path(file.filename).suffix or ".mp4"
    tmp_in  = OUTPUT_DIR / f"input_{id(file)}{suffix}"
    tmp_out = OUTPUT_DIR / f"output_{id(file)}.mp4"

    try:
        content = await file.read()
        tmp_in.write_bytes(content)

        cap = cv2.VideoCapture(str(tmp_in))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_out), fourcc, fps, (width, height))

        frame_idx   = 0
        processed   = 0
        last_faces  = []
        stats = {"total_frames": total, "processed": 0, "faces_detected": 0}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % process_every == 0:
                last_faces = _predict_faces(pipe, frame, threshold)
                processed += 1
                if last_faces:
                    stats["faces_detected"] += len(last_faces)

            annotated = _draw_faces(frame, last_faces, threshold)
            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()

        stats["processed"] = processed

        return {
            "message":   "Video processed successfully.",
            "output_url": f"/api/demo/output/{tmp_out.name}",
            "stats":     stats,
            "filename":  tmp_out.name,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        if tmp_in.exists():
            tmp_in.unlink(missing_ok=True)


@router.get("/api/demo/output/{filename}")
def download_output(filename: str):
    """Serve a processed output video for download / inline playback."""
    # Security: only allow files in OUTPUT_DIR, no path traversal
    safe_name = Path(filename).name
    video_path = OUTPUT_DIR / safe_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=safe_name,
    )
