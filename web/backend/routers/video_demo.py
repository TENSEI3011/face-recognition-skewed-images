"""
video_demo.py — Video processing + WebSocket live stream router

Three modes:
  1. POST /api/demo/process    — Upload video → process → return annotated video URL
  2. WebSocket /ws/stream      — Real-time webcam frame analysis
  3. GET  /api/demo/output/{filename} — Download processed video

Improvements over original:
  - FAISS-first matching in _predict_faces (fastest, open-set aware)
  - Quality gates: skip blurry frames and tiny-face detections
  - Temporal voting in video upload (6/10 majority before confirming identity)
  - Richer stats in API response (confirmed_identity, quality_skipped, etc.)
"""

import cv2
import json
import base64
import asyncio
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse

import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from web.backend.services import pipeline_service
from web.backend.services.temporal_service import TemporalVoter
from web.backend.config import DEFAULT_TOP_K, FAISS_THRESHOLD

router = APIRouter(tags=["demo"])

# Temp dir for processed videos
OUTPUT_DIR = ROOT / "web" / "backend" / "_video_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Quality thresholds ─────────────────────────────────────────────────────────
# Angled faces have fewer high-freq edges visible → lower Laplacian variance naturally.
# H.264-compressed stock/phone video scores lower than raw frames — use 5.0 to
# avoid rejecting valid frames. Truly blurry frames score <2.
MIN_FACE_AREA_PX = 100    # ~10×10 px minimum — catches small/distant/seated faces
MIN_BLUR_SCORE   = 5.0    # Permissive for H.264 compressed video (was 12.0)

# ── WebSocket stream quality ───────────────────────────────────────────────────
WS_JPEG_QUALITY = 60      # Lower = faster round-trip, less lag (was 75)


# ── FFmpeg binary (bundled with imageio-ffmpeg, no system install needed) ──────

def _get_ffmpeg_exe() -> str:
    """Return path to the bundled ffmpeg binary from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # fall back to system ffmpeg if available


def _reencode_to_h264(src: Path, dst: Path) -> bool:
    """Re-encode src (any codec) to dst as H.264 MP4.
    Tries imageio-ffmpeg first, then system ffmpeg, then shutil copy as fallback.
    Returns True on success, False on failure.
    """
    # Build list of ffmpeg candidates to try
    candidates = []
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    candidates.extend(["ffmpeg", "ffmpeg.exe"])   # system PATH fallbacks

    for ffmpeg in candidates:
        try:
            result = subprocess.run(
                [
                    ffmpeg, "-y",
                    "-i", str(src),
                    "-vcodec", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an",
                    str(dst),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
            )
            if result.returncode == 0:
                print(f"[VideoDemo] Re-encoded with {ffmpeg}")
                return True
            else:
                err = result.stderr.decode(errors='replace')[-500:]
                print(f"[VideoDemo] ffmpeg ({ffmpeg}) failed: {err}")
        except FileNotFoundError:
            continue   # This candidate not available
        except Exception as e:
            print(f"[VideoDemo] FFmpeg re-encode error ({ffmpeg}): {e}")
            continue

    # Last resort: copy raw file as-is (mp4v may not play in browser, but better than nothing)
    try:
        import shutil
        shutil.copy2(str(src), str(dst))
        print("[VideoDemo] No ffmpeg available — copied raw mp4v (may not play in browser)")
        return True
    except Exception as e:
        print(f"[VideoDemo] Even raw copy failed: {e}")
        return False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _blur_score(image: np.ndarray) -> float:
    """Return Laplacian variance of image — higher = sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _draw_faces(frame: np.ndarray, faces: list, threshold: float) -> np.ndarray:
    """Draw bounding boxes and large, legible identity labels on the video frame."""
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    for face in faces:
        box   = face.get("box", {})
        x, y, w, h = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
        label = face.get("identity", "UNKNOWN").replace("_", " ")
        conf  = face.get("confidence", 0.0)
        known = face.get("is_known", False)
        color = (0, 210, 90) if known else (30, 30, 230)
        text  = f"{label}  {conf*100:.1f}%"

        # Font scale grows with face size so text is always legible (min 0.9 to max 1.8)
        face_ratio = w / max(fw, 1)
        font_scale = max(0.9, min(1.8, face_ratio * 7.0))
        txt_thick  = 2
        box_thick  = 3

        # Bounding box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, box_thick)

        # Measure text to size the label background
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, font_scale, txt_thick)
        pad   = 8
        bar_h = th + baseline + pad * 2
        bar_y0 = y - bar_h
        bar_y1 = y
        if bar_y0 < 0:          # if no space above, put label below the box
            bar_y0 = y + h
            bar_y1 = y + h + bar_h

        cv2.rectangle(vis, (x, bar_y0), (x + tw + pad * 2, bar_y1), color, -1)

        # Black drop-shadow then white text for maximum contrast
        tx, ty = x + pad, bar_y1 - pad - baseline
        cv2.putText(vis, text, (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), txt_thick + 1, cv2.LINE_AA)
        cv2.putText(vis, text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), txt_thick, cv2.LINE_AA)

    return vis


def _frame_to_b64(frame: np.ndarray, quality: int = WS_JPEG_QUALITY) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _upscale_face_crop(image: np.ndarray, box: list, target_size: int = 112) -> np.ndarray | None:
    """
    Crop and upscale a very small detected face to give ArcFace enough pixels.
    When a face is only 20-30px wide, ArcFace (112x112 input) receives a blurry
    upscale from alignment. Doing the upscale BEFORE alignment with INTER_CUBIC
    gives sharper texture for embedding extraction.
    """
    x, y, w, h = box
    # Add 30% padding to capture full face context
    pad_x = int(w * 0.30)
    pad_y = int(h * 0.30)
    ih, iw = image.shape[:2]
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    # Upscale to at least target_size on the shorter side
    ch, cw = crop.shape[:2]
    if cw < target_size or ch < target_size:
        scale_up = max(target_size / cw, target_size / ch)
        crop = cv2.resize(crop, (int(cw * scale_up), int(ch * scale_up)),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def _predict_faces(
    pipe,
    frame: np.ndarray,
    threshold: float,
    max_side: int = 320,
    use_blur_gate: bool = True,
) -> list:
    """Run detection + recognition on a frame. Returns list of face dicts.

    Parameters
    ----------
    max_side : int
        Longest side of the detection-resolution frame.
        For video: pass 0 or a very large value to use the FULL resolution frame.
        For live webcam: 640 is a good balance of speed vs accuracy.
    use_blur_gate : bool
        Skip the Laplacian blur check on the aligned crop.
        Keep True for live stream (helps filter bad frames).
        Set False for video upload (frames are already sharp enough).

    Recognition priority (best → fallback):
      1. FAISS matcher (in-memory, fastest, open-set aware) — preferred
      2. MongoDB cosine search                               — fallback
      3. SVM pipeline                                       — last resort
    """
    orig_h, orig_w = frame.shape[:2]

    # ── Detection resolution strategy ──────────────────────────────────────────
    # KEY INSIGHT: SCRFD uses an internal 640x640 detection grid.
    # If we pass a 1920x1080 frame, SCRFD internally scales it to fit 640x640,
    # making a 30px face become only ~10px on the detection grid — undetectable.
    #
    # Fix: Do NOT downscale large frames. Pass at full resolution or upscale
    # slightly so that small faces appear larger on SCRFD's internal grid.
    # The pipeline.detector.detect() already handles upscaling for small images
    # (min_side < 300). For large HD frames, we pass them as-is.
    if max_side <= 0 or max_side >= max(orig_w, orig_h):
        # Full resolution — best for detecting small faces in HD video
        small = frame
        scale = 1.0
    else:
        scale = min(max_side / orig_w, max_side / orig_h, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
            scale = 1.0

    # ── Determine recognition path ──────────────────────────────────────────────────
    faiss_matcher = pipeline_service.get_faiss_matcher()
    use_faiss     = faiss_matcher is not None and faiss_matcher.is_built

    from web.backend.services.mongo_service import is_connected
    use_mongo = is_connected() and not use_faiss

    detections = pipe.detector.detect(small)
    results = []

    for det in detections[:4]:
        try:
            bx, by, bw, bh = det["box"]

            # ── Quality gate 1: face box too small ──────────────────────────
            if bw * bh < MIN_FACE_AREA_PX:
                continue

            # ── Scale box back to original frame coordinates ───────────────────
            x = int(bx / scale)
            y = int(by / scale)
            w = int(bw / scale)
            h = int(bh / scale)

            # ── Align from ORIGINAL high-res frame for best ArcFace quality ──
            # Aligning from the small (downscaled) frame loses texture detail.
            # Provide a synthetic detection dict in original coords.
            orig_det = {
                "box":       [x, y, w, h],
                "confidence": det["confidence"],
                "keypoints":  {  # scale keypoints back to original coords
                    kn: (int(kv[0] / scale), int(kv[1] / scale))
                    for kn, kv in det.get("keypoints", {}).items()
                },
            }

            # For very small faces, upscale the crop region first for better quality
            SMALL_FACE_THRESHOLD = 48  # px — faces smaller than this need upscaling
            if w < SMALL_FACE_THRESHOLD or h < SMALL_FACE_THRESHOLD:
                # Upscale the face region, then re-detect within the crop for landmarks
                up_crop = _upscale_face_crop(frame, [x, y, w, h], target_size=112)
                if up_crop is not None:
                    uh, uw = up_crop.shape[:2]
                    # Build a synthetic detection dict for the upscaled crop
                    # Place the face in the center of the upscaled crop
                    up_det = {
                        "box": [0, 0, uw, uh],
                        "confidence": det["confidence"],
                        "keypoints": {  # scale original keypoints into upscaled crop coords
                            kn: (
                                int((kv[0] - (x - int(w * 0.30))) * (uw / max(w + 2 * int(w * 0.30), 1))),
                                int((kv[1] - (y - int(h * 0.30))) * (uh / max(h + 2 * int(h * 0.30), 1))),
                            )
                            for kn, kv in orig_det["keypoints"].items()
                        },
                    }
                    aligned = pipe.aligner.align_from_detection(up_crop, up_det)
                    if aligned is None:
                        aligned = cv2.resize(up_crop, (112, 112), interpolation=cv2.INTER_CUBIC)
                else:
                    aligned = pipe.aligner.align_from_detection(frame, orig_det)
            else:
                aligned = pipe.aligner.align_from_detection(frame, orig_det)

            if aligned is None:
                # Fallback: align from small frame
                aligned = pipe.aligner.align_from_detection(small, det)
            if aligned is None:
                continue

            # ── Quality gate 2: blur check (optional) ───────────────────────
            if use_blur_gate and _blur_score(aligned) < MIN_BLUR_SCORE:
                continue

            # ── Recognition ───────────────────────────────────────────────────
            if use_faiss and pipe.arc_ext:
                emb = pipe.arc_ext.extract(aligned)
                top_label, top_conf = faiss_matcher.best_match(emb)

            elif use_mongo and pipe.arc_ext:
                from web.backend.services.embedding_service import find_best_match
                emb   = pipe.arc_ext.extract(aligned)
                match = find_best_match(emb, threshold=threshold)
                top_label = match["identity"]
                top_conf  = match["confidence"]

            else:
                labels, confs = pipe.identify_from_aligned(aligned, top_k=1)
                top_conf  = float(confs[0]) if confs else 0.0
                top_label = labels[0] if confs and top_conf >= threshold else "UNKNOWN"

            if use_faiss and pipe.arc_ext:
                identity_known = (top_label != "UNKNOWN")
            else:
                identity_known = (top_conf >= threshold)

            results.append({
                "box":        {"x": max(0, x), "y": max(0, y), "w": w, "h": h},
                "identity":   top_label,
                "confidence": round(top_conf, 4),
                "is_known":   identity_known,
            })
        except Exception:
            pass

    return results


# ── WebSocket live stream ──────────────────────────────────────────────────────

@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time face recognition stream with temporal voting.

    Each WebSocket connection has its own TemporalVoter. Predictions are
    only confirmed after the same identity appears in 6/10 consecutive frames,
    eliminating single-frame false positives from blur, occlusion, or compression.

    Quality gates skip blurry frames entirely so they don't pollute the vote.

    Client sends JPEG frames as binary; server responds with JSON:
      { type, faces, frame, temporal_status }
    """
    await websocket.accept()

    status = pipeline_service.get_status()
    if not status["loaded"]:
        await websocket.send_json({"error": "Pipeline not loaded.", "faces": []})
        await websocket.close()
        return

    pipe = pipeline_service.get_pipeline()
    threshold = 0.35

    # One voter per connection — garbage collected when connection closes
    voter = TemporalVoter(window=10, min_votes=6, min_confidence=0.40, cooldown_s=1.5)

    try:
        while True:
            data = await websocket.receive_bytes()

            # Check for JSON control messages (threshold updates, voter reset)
            try:
                msg = json.loads(data.decode("utf-8"))
                if "threshold" in msg:
                    threshold = float(msg["threshold"])
                    await websocket.send_json({"type": "config_ack", "threshold": threshold})
                if msg.get("type") == "reset_voter":
                    voter.reset()
                    await websocket.send_json({"type": "voter_reset"})
                continue
            except Exception:
                pass  # Not JSON — treat as image bytes

            # Decode frame
            arr   = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Quality gate: skip blurry frames entirely (don't even run inference)
            if _blur_score(frame) < MIN_BLUR_SCORE:
                continue

            # Run inference in thread pool to avoid blocking the event loop
            loop  = asyncio.get_running_loop()
            # Use 640px for live stream (was 320 — too aggressive for typical webcams)
            faces = await loop.run_in_executor(
                None, _predict_faces, pipe, frame, threshold, 640)

            # ── Temporal voting ───────────────────────────────────────────
            if faces:
                best_face   = max(faces, key=lambda f: f.get("confidence", 0.0))
                vote_result = voter.update(
                    label=best_face.get("identity", "UNKNOWN"),
                    confidence=best_face.get("confidence", 0.0),
                )
                if vote_result["confirmed"] and vote_result["is_known"]:
                    for f in faces:
                        f["temporal_identity"]  = vote_result["identity"]
                        f["temporal_confirmed"] = True
                else:
                    for f in faces:
                        f["temporal_identity"]  = "UNKNOWN"
                        f["temporal_confirmed"] = False
            else:
                voter.reset()
                vote_result = {"identity": "UNKNOWN", "confirmed": False,
                               "votes": 0, "is_known": False}

            # Use temporally confirmed identity for the response
            display_faces = []
            for f in faces:
                df = dict(f)
                if f.get("temporal_confirmed"):
                    df["identity"] = f["temporal_identity"]
                    df["is_known"] = True
                display_faces.append(df)

            # Send ONLY face JSON — no JPEG frame.
            # The client draws the live webcam directly and overlays these boxes,
            # so we never need to encode/send a server-side annotated frame.
            # This removes ~5–50ms of JPEG encode + base64 overhead per frame.
            response = {
                "type":  "frame",
                "faces": display_faces,
                "temporal_status": {
                    "identity":  vote_result.get("identity",  "UNKNOWN"),
                    "confirmed": vote_result.get("confirmed", False),
                    "votes":     vote_result.get("votes",     0),
                    "window":    voter.window,
                },
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
    file:          UploadFile = File(...),
    threshold:     float      = Form(0.35),
    process_every: int        = Form(3),
):
    """
    Upload a video -> run face recognition -> return annotated video.
    All CPU-heavy work runs in a thread pool so the event loop stays free.
    """
    status = pipeline_service.get_status()
    if not status["loaded"]:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    pipe = pipeline_service.get_pipeline()

    # Use a stable unique ID computed BEFORE reading the file (id() is stable here)
    # NOTE: Do NOT use id(file) after await — Python may GC/reuse the object ID.
    import uuid
    job_id  = uuid.uuid4().hex[:12]
    suffix  = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_in  = OUTPUT_DIR / f"input_{job_id}{suffix}"
    tmp_out = OUTPUT_DIR / f"output_{job_id}.mp4"
    tmp_raw = OUTPUT_DIR / f"raw_{job_id}.mp4"

    # Read uploaded bytes in async context (fast — just reading from memory)
    content = await file.read()
    tmp_in.write_bytes(content)

    # ── Run ALL CPU-heavy processing in a background thread ──────────────────
    # This is critical: cv2 frame processing + face detection + writing video
    # are all blocking CPU operations. Running them in the async event loop
    # blocks uvicorn and causes connection aborts on long videos.
    import asyncio
    loop = asyncio.get_running_loop()

    def _process_sync():
        """Synchronous video processing — runs in thread pool."""
        cap = cv2.VideoCapture(str(tmp_in))
        if not cap.isOpened():
            raise RuntimeError("Could not open video file.")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_raw), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not initialise video writer.")

        voter = TemporalVoter(
            window=10,       # 10-frame rolling window (was 15)
            min_votes=5,     # 5/10 majority needed (was 8/15) — faster confirmation
            min_confidence=0.35,
            cooldown_s=0.0,
        )

        frame_idx       = 0
        processed       = 0
        quality_skipped = 0
        last_faces      = []
        confirmed_id    = "UNKNOWN"
        confirmed_known = False
        faces_detected  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % process_every == 0:
                if _blur_score(frame) < MIN_BLUR_SCORE:
                    quality_skipped += 1
                else:
                    # Use full resolution for video — do NOT downscale.
                    # SCRFD uses an internal 640x640 detection grid. If we pass a
                    # 1920x1080 frame downscaled to 1280px, faces that are only
                    # 20-30px in the original become ~10px on the internal grid
                    # and are missed entirely. max_side=0 means full resolution.
                    raw_faces = _predict_faces(
                        pipe, frame, threshold,
                        max_side=0,          # 0 = use full resolution, no downscaling
                        use_blur_gate=False,
                    )
                    processed += 1

                    if raw_faces:
                        faces_detected += len(raw_faces)
                        best = max(raw_faces, key=lambda f: f.get("confidence", 0.0))
                        vote = voter.update(
                            label=best.get("identity", "UNKNOWN"),
                            confidence=best.get("confidence", 0.0),
                        )
                        if vote["confirmed"] and vote["is_known"]:
                            confirmed_id    = vote["identity"]
                            confirmed_known = True
                        else:
                            # Not confirmed yet OR person is unknown.
                            # Show "UNKNOWN" — not "PENDING" (PENDING is confusing to users).
                            # PENDING means: voter window not full yet, just say UNKNOWN.
                            confirmed_id    = "UNKNOWN"
                            confirmed_known = False

                        last_faces = []
                        for f in raw_faces:
                            fd = dict(f)
                            fd["identity"] = confirmed_id
                            fd["is_known"]  = confirmed_known
                            last_faces.append(fd)
                    else:
                        voter.reset()
                        confirmed_id    = "UNKNOWN"
                        confirmed_known = False
                        last_faces = []

            annotated = _draw_faces(frame, last_faces, threshold)
            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()

        return {
            "total_frames":       total,
            "processed":          processed,
            "quality_skipped":    quality_skipped,
            "faces_detected":     faces_detected,
            "confirmed_identity": confirmed_id if confirmed_known else "UNKNOWN",
        }

    try:
        # Run in thread pool — will NOT block event loop
        stats = await loop.run_in_executor(None, _process_sync)

        # Re-encode raw mp4v -> H264 for browser playback (also in thread)
        ok = await loop.run_in_executor(None, _reencode_to_h264, tmp_raw, tmp_out)
        if not ok:
            tmp_raw.rename(tmp_out)
        else:
            tmp_raw.unlink(missing_ok=True)   # clean up intermediate

        return {
            "message":    "Video processed successfully.",
            "output_url": f"/api/demo/output/{tmp_out.name}",
            "stats":      stats,
            "filename":   tmp_out.name,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[VideoDemo] process_video exception:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        if tmp_in.exists():
            tmp_in.unlink(missing_ok=True)
        # Clean up raw intermediate (use job_id — stable, not id(file))
        if tmp_raw.exists():
            tmp_raw.unlink(missing_ok=True)


@router.get("/api/demo/output/{filename}")
def download_output(filename: str):
    """Serve a processed output video for inline playback or download."""
    safe_name  = Path(filename).name
    video_path = OUTPUT_DIR / safe_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{safe_name}"',
            "Accept-Ranges":       "bytes",
            "Cache-Control":       "no-cache, no-store, must-revalidate",
            "Pragma":              "no-cache",
            "Expires":             "0",
            "X-Content-Type-Options": "nosniff",
        },
    )
