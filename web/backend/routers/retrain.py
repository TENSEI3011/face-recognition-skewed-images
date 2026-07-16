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
from web.backend.config   import GALLERY_DIR, MODELS_DIR, FAISS_THRESHOLD

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


def _extract_feat_from_image(pipe, img: np.ndarray):
    """
    Attempt to extract a feature vector from img using two strategies:
      1. Direct _extract_from_aligned (for pre-cropped 112×112 chips)
      2. Full detect → align → extract fallback
    Returns None if both strategies fail.
    """
    feat = None

    # Strategy 1: treat as pre-cropped face chip — direct extraction
    try:
        feat = pipe._extract_from_aligned(img)
        if feat is not None and pipe.arc_ext is not None:
            arc_part = feat[:512] if len(feat) >= 512 else feat
            if np.linalg.norm(arc_part) < 1e-6:
                feat = None  # ArcFace returned zero — try fallback
    except Exception:
        feat = None

    # Strategy 2: fallback — detect → align → extract
    if feat is None:
        try:
            detection = pipe.detector.detect_largest(img)
            if detection is not None:
                aligned = pipe.aligner.align_from_detection(img, detection)
                if aligned is not None:
                    feat = pipe._extract_from_aligned(aligned)
        except Exception:
            feat = None

    return feat


def _load_gallery_features(pipe, job_id: str):
    """
    Smart gallery feature extractor.

    Handles both:
      - Pre-cropped 112×112 face chips (most gallery images)
      - Full photos with face detection fallback

    Gallery augmentation:
      For each successfully extracted original image, ALSO extract features
      from MILD and MODERATE UAV-degraded copies. This 3× the training data
      with zero new photos — giving SVM a much more robust view of each person
      under different lighting and blur conditions (e.g. outdoor/balcony video).

      WHY THIS HELPS FOR VIDEO IDENTIFICATION:
        Gallery photos are usually taken indoors or close-up (clean).
        Probe frames from outdoor videos have UAV-like degradation.
        Training on augmented gallery makes SVM generalize across this gap.
    """
    # Pre-build augmentors once (avoids rebuilding albumentations pipeline per image)
    try:
        from src.augmentation import UAVAugmentor
        aug_mild     = UAVAugmentor(profile="mild",     seed=42)
        aug_moderate = UAVAugmentor(profile="moderate", seed=42)
        use_augment  = True
    except Exception as _ae:
        _log(job_id, f"  [Augmentation] UAVAugmentor unavailable ({_ae}) — skipping augmentation.")
        aug_mild = aug_moderate = None
        use_augment = False

    X, y = [], []
    identity_dirs = sorted([d for d in GALLERY_DIR.iterdir() if d.is_dir()])

    for id_dir in identity_dirs:
        identity   = id_dir.name
        image_files = [
            f for f in id_dir.iterdir()
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        saved   = 0
        skipped = 0
        aug_saved = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            # ── Original image ─────────────────────────────────────────────
            feat = _extract_feat_from_image(pipe, img)
            if feat is not None:
                X.append(feat)
                y.append(identity)
                saved += 1

                # ── Augmented variants (only if original succeeded) ────────
                if use_augment:
                    for aug, profile_name in [(aug_mild, "mild"), (aug_moderate, "moderate")]:
                        try:
                            aug_img  = aug.augment(img)
                            aug_feat = _extract_feat_from_image(pipe, aug_img)
                            if aug_feat is not None:
                                X.append(aug_feat)
                                y.append(identity)
                                aug_saved += 1
                        except Exception:
                            pass  # Augmentation failed — not critical
            else:
                skipped += 1

        aug_info = f" + {aug_saved} augmented" if use_augment else ""
        _log(job_id, f"  {identity}: {saved} original{aug_info}, {skipped} skipped.")

    _log(job_id, f"Total: {len(X)} feature vectors from {len(set(y))} identities "
                 f"({'with' if use_augment else 'without'} augmentation).")

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
        # X contains the raw feature vectors before PCA. We only need the first
        # 512 dimensions (ArcFace slice) for cosine / FAISS matching.
        import numpy as _np
        gallery_arc = {}
        arc_embeddings = []   # flat list for FAISS index
        arc_labels     = []   # parallel identity list for FAISS

        for feat, identity in zip(X, y):
            norm = _np.linalg.norm(feat)
            if norm > 1e-6:
                normed = feat / norm
                gallery_arc.setdefault(identity, []).append(normed)
                # ArcFace is always the last 512 dims when 'arcface' is a modality.
                # Grab the raw (un-normalized) ArcFace slice for FAISS.
                arc_slice = feat[-512:] if len(feat) >= 512 else feat
                arc_embeddings.append(arc_slice.astype(_np.float32))
                arc_labels.append(identity)

        pipeline_service.set_gallery_arc_features(gallery_arc)
        n_stored = sum(len(v) for v in gallery_arc.values())
        _log(job_id, f"Stored {n_stored} ArcFace embeddings ({len(gallery_arc)} identities) for cosine rejection.")

        # Build FAISS index with raw ArcFace embeddings extracted directly from
        # gallery images. This MUST match what identify.py uses at inference:
        #   pipe.arc_ext.extract(aligned_q)  → raw 512-dim ArcFace embedding
        # Do NOT use slices from the fused feature vector (different space).
        try:
            from src.matcher import FAISSMatcher
            if pipe.arc_ext is not None:
                arc_embeddings_raw = []
                arc_labels_raw     = []

                for id_dir in sorted([d for d in GALLERY_DIR.iterdir() if d.is_dir()]):
                    identity    = id_dir.name
                    image_files = [f for f in id_dir.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS]
                    for img_path in image_files:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        # Use pre-crop fast path for small images
                        h, w = img.shape[:2]
                        if max(h, w) <= 150:
                            face = cv2.resize(img, (112, 112))
                        else:
                            det = pipe.detector.detect_largest(img)
                            face = pipe.aligner.align_from_detection(img, det) if det else None
                            if face is None:
                                face = cv2.resize(img, (112, 112))
                        try:
                            emb = pipe.arc_ext.extract(face)   # raw 512-dim
                            arc_embeddings_raw.append(emb.astype(_np.float32))
                            arc_labels_raw.append(identity)
                        except Exception:
                            pass

                if arc_embeddings_raw:
                    matcher = FAISSMatcher(threshold=FAISS_THRESHOLD, dim=512)
                    emb_matrix = _np.stack(arc_embeddings_raw, axis=0)
                    matcher.add_gallery(emb_matrix, arc_labels_raw)
                    matcher.save(str(MODELS_DIR / "faiss_matcher"))
                    pipeline_service.set_faiss_matcher(matcher)
                    _log(job_id, f"FAISS index built: {matcher.n_gallery} raw ArcFace vectors, "
                                 f"{matcher.n_identities} identities, threshold={FAISS_THRESHOLD}")
                else:
                    _log(job_id, "No ArcFace embeddings extracted — FAISS index skipped.")
            else:
                _log(job_id, "ArcFace extractor not in pipeline — FAISS index skipped.")
        except ImportError:
            _log(job_id, "faiss-cpu not installed — using cosine loop fallback. "
                         "Run: pip install faiss-cpu   to enable FAISS.")
        except Exception as _fe:
            _log(job_id, f"FAISS build failed (non-critical): {_fe}")


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
