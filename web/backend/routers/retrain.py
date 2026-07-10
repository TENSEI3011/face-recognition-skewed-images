"""
retrain.py - Hot-retrain the face recognition pipeline from the gallery.

POST /api/pipeline/retrain
    Starts a background job that:
      1. Loads all gallery images and extracts features
         (handles pre-cropped 112x112 face chips AND full photos)
      2. Trains PCA + SVM on the new gallery
      3. Saves the model to disk
      4. Hot-swaps the in-memory pipeline (live stream picks it up instantly)
    Returns a job_id that the frontend can poll.

GET /api/pipeline/retrain/status/{job_id}
    Returns the current status + log output of the retrain job.
"""

import sys
import cv2
import uuid
import numpy as np
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

# Project root on path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from web.backend.services import pipeline_service
from web.backend.config   import GALLERY_DIR, MODELS_DIR

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# In-memory job store
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _make_job() -> str:
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            "id":       job_id,
            "status":   "pending",   # pending | running | done | error
            "log":      [],
            "started":  None,
            "finished": None,
            "error":    None,
        }
    return job_id


def _log(job_id: str, msg: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _jobs_lock:
        _jobs[job_id]["log"].append(line)
    try:
        print(line)
    except Exception:
        pass  # Suppress encoding errors on Windows stdout


def _load_gallery_features(pipe, job_id: str):
    """
    Smart gallery feature extractor that handles both:
      - Pre-cropped 112x112 face chips (most gallery images)
      - Full photos with face detection fallback

    Strategy per image:
      1. Try _extract_from_aligned() directly (treats whole image as face chip)
      2. If zero result, fallback to detect -> align -> extract
      3. Skip if both fail
    """
    X, y = [], []
    identity_dirs = sorted([d for d in GALLERY_DIR.iterdir() if d.is_dir()])

    for id_dir in identity_dirs:
        identity = id_dir.name
        image_files = [
            f for f in id_dir.iterdir()
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        saved = 0
        skipped = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            feat = None

            # Strategy 1: treat as pre-cropped face chip - direct feature extraction
            try:
                feat = pipe._extract_from_aligned(img)
                # Check if ArcFace returned a zero vector (fails on some image types)
                if feat is not None and pipe.arc_ext is not None:
                    arc_part = feat[:512] if len(feat) >= 512 else feat
                    if np.linalg.norm(arc_part) < 1e-6:
                        feat = None  # Zero result - try fallback
            except Exception:
                feat = None

            # Strategy 2: fallback - full detect -> align -> extract
            if feat is None:
                try:
                    detection = pipe.detector.detect_largest(img)
                    if detection is not None:
                        aligned = pipe.aligner.align_from_detection(img, detection)
                        if aligned is not None:
                            feat = pipe._extract_from_aligned(aligned)
                except Exception:
                    feat = None

            if feat is not None:
                X.append(feat)
                y.append(identity)
                saved += 1
            else:
                skipped += 1

        _log(job_id, f"  {identity}: {saved} features extracted, {skipped} skipped.")

    return np.array(X, dtype=np.float32), y


def _run_retrain(job_id: str) -> None:
    """
    Background worker: re-train PCA + SVM from gallery, save to disk,
    then hot-swap the in-memory pipeline singleton.
    """
    with _jobs_lock:
        _jobs[job_id]["status"]  = "running"
        _jobs[job_id]["started"] = datetime.utcnow().isoformat()

    try:
        _log(job_id, "Retrain started.")

        # Validate gallery
        if not GALLERY_DIR.exists():
            raise RuntimeError(f"Gallery directory not found: {GALLERY_DIR}")

        identity_dirs = [d for d in GALLERY_DIR.iterdir() if d.is_dir()]
        if len(identity_dirs) < 2:
            raise RuntimeError(
                f"Need at least 2 identities in the gallery to train. "
                f"Found: {len(identity_dirs)}"
            )
        _log(job_id, f"Gallery has {len(identity_dirs)} identities.")

        # Get a pipeline instance (reuse loaded models for feature extractors)
        pipe = pipeline_service.get_pipeline()
        if pipe is None:
            raise RuntimeError(
                "No pipeline loaded. Run the baseline experiment first to "
                "initialize the pipeline models."
            )

        # Extract features using smart loader (handles pre-cropped chips)
        _log(job_id, "Extracting features from gallery images...")
        X, y = _load_gallery_features(pipe, job_id)

        if len(X) == 0:
            raise RuntimeError(
                "Could not extract features from any gallery image. "
                "Check that gallery images contain visible faces."
            )

        _log(job_id, f"Extracted {len(X)} feature vectors from {len(set(y))} identities.")

        # Train PCA + SVM
        _log(job_id, "Fitting PCA reducer...")
        pipe.train(X, y)
        _log(job_id, "SVM classifier trained successfully.")

        # Save to disk
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pipe.save(str(MODELS_DIR))
        _log(job_id, f"Model saved to disk.")

        # Store per-class ArcFace embeddings in memory for cosine rejection.
        # X contains the raw feature vectors (512-dim ArcFace) before PCA.
        # These are extracted with the SAME method used at inference time,
        # so cosine similarity against these is reliable.
        import numpy as _np
        gallery_arc = {}
        for feat, identity in zip(X, y):
            norm = _np.linalg.norm(feat)
            if norm > 1e-6:
                gallery_arc.setdefault(identity, []).append(feat / norm)
        pipeline_service.set_gallery_arc_features(gallery_arc)
        n_stored = sum(len(v) for v in gallery_arc.values())
        _log(job_id, f"Stored {n_stored} ArcFace embeddings ({len(gallery_arc)} identities) for cosine rejection.")

        # Hot-swap the singleton so the live stream picks up immediately
        pipeline_service._status["loaded"] = True
        pipeline_service._status["error"]  = None

        _log(job_id, "[DONE] Pipeline retrained and hot-swapped. Live stream updated immediately.")

        with _jobs_lock:
            _jobs[job_id]["status"]   = "done"
            _jobs[job_id]["finished"] = datetime.utcnow().isoformat()

    except Exception as exc:
        _log(job_id, f"[ERROR] {str(exc)}")
        with _jobs_lock:
            _jobs[job_id]["status"]   = "error"
            _jobs[job_id]["error"]    = str(exc)
            _jobs[job_id]["finished"] = datetime.utcnow().isoformat()


# Endpoints

@router.post("/retrain")
def start_retrain():
    """
    Kick off a background retrain from the current gallery.
    Returns immediately with a job_id. Poll /retrain/status/{job_id} for updates.
    """
    # Prevent two retrains running simultaneously
    with _jobs_lock:
        running = [j for j in _jobs.values() if j["status"] == "running"]
    if running:
        return {
            "message": "A retrain is already in progress.",
            "job_id":  running[0]["id"],
            "already_running": True,
        }

    job_id = _make_job()
    t = threading.Thread(target=_run_retrain, args=(job_id,), daemon=True)
    t.start()

    return {
        "message":         "Retrain started in the background.",
        "job_id":          job_id,
        "already_running": False,
    }


@router.get("/retrain/status/{job_id}")
def retrain_status(job_id: str):
    """Poll the status and log output of a retrain job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {
        "id":       job["id"],
        "status":   job["status"],
        "log":      job["log"],
        "started":  job["started"],
        "finished": job["finished"],
        "error":    job["error"],
    }


@router.get("/retrain/jobs")
def list_retrain_jobs():
    """List all retrain jobs (for debugging)."""
    with _jobs_lock:
        return {"jobs": list(_jobs.values())}
