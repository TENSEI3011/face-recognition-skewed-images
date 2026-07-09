"""
gallery.py — Gallery management router

PRIMARY:  MongoDB Atlas — stores ArcFace embeddings + base64 thumbnails
FALLBACK: Local disk /data/gallery/ (if MongoDB not connected)

Endpoints:
  GET    /api/gallery                → list all enrolled identities
  GET    /api/gallery/{name}/images  → get thumbnail images for identity
  POST   /api/gallery/upload         → upload images, extract & store embeddings
  DELETE /api/gallery/{name}         → remove identity from gallery
"""

import cv2
import base64
import shutil
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
    name:  str = Form(...),
    files: List[UploadFile] = File(...),
):
    """
    Upload images for an identity.
    - If MongoDB is connected: extract ArcFace embedding → store in MongoDB
    - If not connected: save to local disk (original behaviour)
    """
    name = name.strip().replace(" ", "_")
    if not name:
        raise HTTPException(status_code=400, detail="Identity name cannot be empty.")
    if any(c in name for c in r'\/:*?"<>|'):
        raise HTTPException(status_code=400, detail="Invalid characters in identity name.")

    saved  = []
    errors = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            errors.append(f"{upload.filename}: unsupported format")
            continue

        try:
            content = await upload.read()

            if is_connected():
                # ── MongoDB path: extract embedding → store ──────────────────
                arr   = np.frombuffer(content, dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image is None:
                    errors.append(f"{upload.filename}: could not decode image")
                    continue

                emb = _extract_embedding(image)
                if emb is None:
                    errors.append(f"{upload.filename}: no face detected — skipped")
                    continue

                thumb_b64 = _image_to_b64_thumbnail(image)
                ok = store_embedding(
                    identity=name,
                    embedding=emb,
                    image_b64=thumb_b64,
                    filename=upload.filename,
                )
                if ok:
                    saved.append(upload.filename)
                else:
                    errors.append(f"{upload.filename}: failed to store in MongoDB")

            else:
                # ── Local fallback: save to disk ─────────────────────────────
                id_dir = GALLERY_DIR / name
                id_dir.mkdir(parents=True, exist_ok=True)
                dest = id_dir / upload.filename
                counter = 1
                while dest.exists():
                    dest = id_dir / f"{Path(upload.filename).stem}_{counter}{suffix}"
                    counter += 1
                dest.write_bytes(content)
                saved.append(upload.filename)

        except Exception as e:
            errors.append(f"{upload.filename}: {e}")

    storage = "MongoDB Atlas" if is_connected() else "local disk"
    return {
        "identity": name,
        "saved":    saved,
        "errors":   errors,
        "message":  f"Uploaded {len(saved)} image(s) for '{name}' to {storage}.",
        "embeddings_stored": is_connected(),
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
