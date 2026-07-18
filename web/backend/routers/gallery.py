"""
gallery.py — Gallery management router

PRIMARY:  MongoDB Atlas — stores ArcFace embeddings + base64 thumbnails
FALLBACK: Local disk /data/gallery/ (if MongoDB not connected)

KEY DESIGN:
  Images are ALWAYS saved to disk regardless of MongoDB status.
  This ensures retrain.py (which reads GALLERY_DIR) always sees every
  enrolled identity — fixing the UNKNOWN bug after MongoDB enrollment.

  A background retrain is triggered automatically after every upload
  so users never need to click "Retrain from Gallery" manually.

Endpoints:
  GET    /api/gallery                → list all enrolled identities
  GET    /api/gallery/{name}/images  → get thumbnail images for identity
  POST   /api/gallery/upload         → upload images, extract & store embeddings
  DELETE /api/gallery/{name}         → remove identity from gallery
"""

import cv2
import base64
import shutil
import threading
import numpy as np
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from web.backend.config import GALLERY_DIR
from web.backend.services.mongo_service import get_db, is_connected
from web.backend.services.embedding_service import (
    store_embedding, delete_identity, list_identities, get_identity_images
)

router = APIRouter(prefix="/api/gallery", tags=["gallery"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Minimum Laplacian variance for an aligned face to be accepted for enrollment.
# Faces below this threshold are too blurry to produce reliable ArcFace embeddings.
# Typical values: sharp face chip ~100-500, blurry ~5-30, unusable <10.
ENROLL_MIN_BLUR_SCORE = 40.0



# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_arcface():
    """Get the ArcFace extractor from the loaded pipeline (reuses loaded model)."""
    try:
        from web.backend.services import pipeline_service
        pipe = pipeline_service.get_pipeline()
        if pipe and pipe.arc_ext:
            return pipe.arc_ext
    except Exception:
        pass
    return None


def _extract_embedding(image: np.ndarray) -> np.ndarray | None:
    """Detect face, align it, extract ArcFace 512-D embedding."""
    try:
        from web.backend.services import pipeline_service
        pipe = pipeline_service.get_pipeline()
        if pipe is None:
            return None
        detection = pipe.detector.detect_largest(image)
        if detection is None:
            return None
        aligned = pipe.aligner.align_from_detection(image, detection)
        if aligned is None:
            return None
        if pipe.arc_ext:
            emb = pipe.arc_ext.extract(aligned)
            return emb
    except Exception as e:
        print(f"[Gallery] Embedding extraction error: {e}")
    return None


def _face_blur_score(image: np.ndarray) -> float:
    """
    Return the Laplacian variance of the image — a simple sharpness metric.
    Higher = sharper. Used as a quality gate at enrollment time.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _extract_embedding_checked(
    image: np.ndarray,
    min_blur: float = ENROLL_MIN_BLUR_SCORE,
) -> tuple[np.ndarray | None, str | None]:
    """
    Like _extract_embedding() but also runs a blur quality gate.

    Returns
    -------
    (embedding, None)      — success
    (None, reason_string)  — failure with a human-readable reason
    """
    try:
        from web.backend.services import pipeline_service
        pipe = pipeline_service.get_pipeline()
        if pipe is None:
            return None, "pipeline not loaded"

        detection = pipe.detector.detect_largest(image)
        if detection is None:
            return None, "no face detected"

        aligned = pipe.aligner.align_from_detection(image, detection)
        if aligned is None:
            return None, "face alignment failed"

        # ── Blur quality gate ─────────────────────────────────────────────
        blur = _face_blur_score(aligned)
        if blur < min_blur:
            return None, f"too blurry (sharpness={blur:.1f} < {min_blur})"

        if pipe.arc_ext:
            emb = pipe.arc_ext.extract(aligned)
            return emb, None
        return None, "ArcFace extractor not available"
    except Exception as e:
        print(f"[Gallery] Embedding extraction error: {e}")
        return None, str(e)



def _extract_embeddings_from_video(
    content: bytes,
    filename: str,
    frame_interval: int = 15,
    max_frames: int = 200,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract ArcFace embeddings from a video file.

    Samples one frame every `frame_interval` frames (default 30 ≈ 1 fps
    for a 30 fps video). Skips frames where no face is detected.

    Returns:
        List of (embedding, face_bgr_image) tuples for every frame
        that yielded a valid face embedding. Empty list on failure.
    """
    import tempfile, os

    results: list[tuple[np.ndarray, np.ndarray]] = []

    # Write bytes to a temp file so OpenCV can open it
    suffix = Path(filename).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        print(f"[Gallery/Video] Could not write temp file: {e}")
        return results

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            print(f"[Gallery/Video] Could not open video: {filename}")
            return results

        frame_idx = 0
        sampled   = 0

        while sampled < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                emb, _ = _extract_embedding_checked(frame)
                if emb is not None:
                    thumb = _image_to_b64_thumbnail(frame)
                    results.append((emb, thumb))

                sampled += 1

            frame_idx += 1

        cap.release()
        print(f"[Gallery/Video] {filename}: sampled {frame_idx} frames, "
              f"got {len(results)} face embeddings.")
    except Exception as e:
        print(f"[Gallery/Video] Error processing {filename}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return results


def _image_to_b64_thumbnail(image: np.ndarray, max_size: int = 200) -> str:
    """Resize and encode image as base64 JPEG thumbnail."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _get_local_identities() -> List[dict]:
    """Local fallback: list identities from disk."""
    identities = []
    if not GALLERY_DIR.exists():
        return identities
    for d in sorted(GALLERY_DIR.iterdir()):
        if d.is_dir():
            images = [f for f in d.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS]
            identities.append({
                "name":        d.name,
                "image_count": len(images),
                "path":        str(d),
            })
    return identities


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
def list_gallery():
    """List all enrolled identities with image counts."""
    if is_connected():
        identities = list_identities()
        return {"identities": identities, "total": len(identities), "source": "mongodb"}
    # Fallback
    identities = _get_local_identities()
    return {"identities": identities, "total": len(identities), "source": "local"}


@router.get("/{name}/images")
def get_gallery_images(name: str, max_images: int = 8):
    """Return base64-encoded thumbnails for an identity."""
    if is_connected():
        images = get_identity_images(name, max_images)
        if not images:
            # Check if identity exists at all
            identities = [i["name"] for i in list_identities()]
            if name not in identities:
                raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
        return {"name": name, "images": images}

    # Local fallback
    id_dir = GALLERY_DIR / name
    if not id_dir.exists():
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
    image_files = sorted([
        f for f in id_dir.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS
    ])[:max_images]
    result = []
    for img_path in image_files:
        try:
            data = img_path.read_bytes()
            b64  = base64.b64encode(data).decode()
            ext  = img_path.suffix.lstrip(".").lower()
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            result.append({"filename": img_path.name, "data": f"data:image/{mime};base64,{b64}"})
        except Exception:
            pass
    return {"name": name, "images": result}


@router.post("/upload")
async def upload_images(
    name:          str            = Form(...),
    files:         List[UploadFile] = File(...),
    frame_interval: int            = Form(15),
):
    """
    Upload images **or videos** for an identity.

    Supported image formats: JPG, JPEG, PNG, BMP, WEBP
    Supported video formats: MP4, AVI, MOV, MKV, WEBM

    For videos: frames are sampled every `frame_interval` frames (default 30).
    Each sampled frame is face-detected and embedded independently.

    - If MongoDB is connected: extract ArcFace embedding → store in MongoDB
    - If not connected: images saved to local disk (videos require MongoDB)
    """
    name = name.strip().replace(" ", "_")
    if not name:
        raise HTTPException(status_code=400, detail="Identity name cannot be empty.")
    if any(c in name for c in r'\/:*?"<>|'):
        raise HTTPException(status_code=400, detail="Invalid characters in identity name.")

    saved  = []   # filenames (or "video_frame_{n}" entries) successfully stored
    errors = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        is_video = suffix in ALLOWED_VIDEO_EXTENSIONS
        is_image = suffix in ALLOWED_EXTENSIONS

        if not is_image and not is_video:
            errors.append(f"{upload.filename}: unsupported format")
            continue

        try:
            content = await upload.read()

            # ── Video path ────────────────────────────────────────────────────
            if is_video:
                pairs = _extract_embeddings_from_video(
                    content, upload.filename, frame_interval=frame_interval
                )
                if not pairs:
                    errors.append(
                        f"{upload.filename}: no faces detected in any sampled frame"
                    )
                    continue

                # Always save face chip thumbnails to disk so retrain sees them
                id_dir = GALLERY_DIR / name
                id_dir.mkdir(parents=True, exist_ok=True)

                frame_saved = 0
                for idx, (emb, thumb_b64) in enumerate(pairs):
                    frame_name = f"{Path(upload.filename).stem}_frame{idx:04d}.jpg"

                    # ── Save JPEG thumbnail to disk (retrain reads this) ───────
                    try:
                        # Decode the base64 thumbnail back to bytes and write
                        b64_data  = thumb_b64.split(",", 1)[-1]  # strip data:image/jpeg;base64,
                        img_bytes = base64.b64decode(b64_data)
                        dest = id_dir / frame_name
                        dest.write_bytes(img_bytes)
                    except Exception as _de:
                        print(f"[Gallery/Video] Could not save frame to disk: {_de}")

                    # ── Also store in MongoDB when connected ───────────────────
                    if is_connected():
                        ok = store_embedding(
                            identity=name,
                            embedding=emb,
                            image_b64=thumb_b64,
                            filename=frame_name,
                        )
                        if not ok:
                            errors.append(f"{frame_name}: failed to store in MongoDB")

                    frame_saved += 1

                saved.append(
                    f"{upload.filename} ({frame_saved} frames enrolled)"
                )
                continue

            # ── Image path ────────────────────────────────────────────────────
            arr   = np.frombuffer(content, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                errors.append(f"{upload.filename}: could not decode image")
                continue

            emb, reason = _extract_embedding_checked(image)
            if emb is None:
                errors.append(f"{upload.filename}: {reason} — skipped")
                continue

            # ── ALWAYS save to disk (retrain.py reads GALLERY_DIR) ────────────
            # This is the fix for the UNKNOWN bug: previously, when MongoDB was
            # connected, images were stored ONLY in MongoDB. But retrain.py only
            # reads from GALLERY_DIR on disk, so new identities were invisible.
            id_dir = GALLERY_DIR / name
            id_dir.mkdir(parents=True, exist_ok=True)
            dest    = id_dir / upload.filename
            counter = 1
            while dest.exists():
                dest = id_dir / f"{Path(upload.filename).stem}_{counter}{suffix}"
                counter += 1
            dest.write_bytes(content)

            # ── Also store embedding in MongoDB when connected ─────────────────
            if is_connected():
                thumb_b64 = _image_to_b64_thumbnail(image)
                ok = store_embedding(
                    identity=name,
                    embedding=emb,
                    image_b64=thumb_b64,
                    filename=upload.filename,
                )
                if not ok:
                    errors.append(f"{upload.filename}: failed to store in MongoDB")

            saved.append(upload.filename)

        except Exception as e:
            errors.append(f"{upload.filename}: {e}")

    storage = "MongoDB + disk" if is_connected() else "local disk"
    total_enrolled = len(saved)

    # ── AUTO-RETRAIN in background ────────────────────────────────────────────
    # Trigger a retrain automatically so the new identity is immediately
    # recognised — no need to manually click "Retrain from Gallery".
    if total_enrolled > 0:
        try:
            from web.backend.routers.retrain import _make_job, _run_retrain
            job_id = _make_job()
            t = threading.Thread(
                target=_run_retrain,
                args=(job_id,),
                daemon=True,
                name=f"auto-retrain-{name}",
            )
            t.start()
            retrain_job_id = job_id
            print(f"[Gallery] Auto-retrain started (job {job_id}) after enrolling '{name}'")
        except Exception as _re:
            print(f"[Gallery] Auto-retrain could not start: {_re}")
            retrain_job_id = None
    else:
        retrain_job_id = None

    return {
        "identity":         name,
        "saved":            saved,
        "errors":           errors,
        "message":          (
            f"Enrolled {total_enrolled} file(s) for '{name}' to {storage}. "
            + ("Model retraining started automatically." if retrain_job_id else "")
        ),
        "embeddings_stored": is_connected(),
        "retrain_job_id":    retrain_job_id,
    }


@router.delete("/{name}")
def remove_identity(name: str):
    """Remove an identity and all its images/embeddings from the gallery."""
    if is_connected():
        count = delete_identity(name)
        if count == 0:
            raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
        return {"message": f"Identity '{name}' deleted ({count} embeddings removed)."}

    # Local fallback
    id_dir = GALLERY_DIR / name
    if not id_dir.exists():
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
    shutil.rmtree(id_dir)
    return {"message": f"Identity '{name}' deleted successfully."}
