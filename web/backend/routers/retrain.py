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
from src.embedding_cache  import get_cached_feature, save_cached_feature, cache_stats

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
        aug_severe   = UAVAugmentor(profile="severe",   seed=42)
        # Extra augmentors with different seeds for small galleries (imbalance fix)
        aug_mild_b   = UAVAugmentor(profile="mild",     seed=7)
        aug_mild_c   = UAVAugmentor(profile="mild",     seed=99)
        use_augment  = True
    except Exception as _ae:
        _log(job_id, f"  [Augmentation] UAVAugmentor unavailable ({_ae}) — skipping augmentation.")
        aug_mild = aug_moderate = aug_severe = None
        aug_mild_b = aug_mild_c = None
        use_augment = False

    X, y = [], []
    identity_dirs = sorted([d for d in GALLERY_DIR.iterdir() if d.is_dir()])

    # Determine active modalities for cache keying
    pipe = pipeline_service.get_pipeline()
    active_modalities = list(pipe.modalities) if pipe else ["hog", "lbp", "geometry", "arcface"]

    # Log cache stats before extraction
    stats = cache_stats(GALLERY_DIR, active_modalities)
    _log(job_id, f"Cache stats before retrain: {stats['cached']}/{stats['total']} images cached "
                 f"(hit-rate {stats['hit_rate']}). "
                 f"Only new/changed images will be re-extracted.")

    # Count images per identity first so we can apply extra augmentation
    # to identities that have fewer images (e.g. enrolled via short video).
    img_counts = {}
    for id_dir in identity_dirs:
        n = len([f for f in id_dir.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS])
        img_counts[id_dir.name] = n
    max_count = max(img_counts.values()) if img_counts else 1

    for id_dir in identity_dirs:
        identity   = id_dir.name
        image_files = [
            f for f in id_dir.iterdir()
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        saved   = 0
        skipped = 0
        aug_saved = 0
        cache_hits = 0

        # Identities with fewer than half the max image count get extra augmentation
        few_images = len(image_files) < max(8, max_count // 2)

        for img_path in image_files:
            # ── Try embedding cache first (fast path) ─────────────────────────
            cached_feat = get_cached_feature(img_path, active_modalities)
            if cached_feat is not None:
                X.append(cached_feat)
                y.append(identity)
                saved += 1
                cache_hits += 1
                # Still run augmentation on cached originals so SVM has varied data
                if use_augment:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        aug_list = [
                            (aug_mild,     "mild"),
                            (aug_moderate, "moderate"),
                            (aug_severe,   "severe"),
                        ]
                        if few_images:
                            aug_list += [(aug_mild_b, "mild_b"), (aug_mild_c, "mild_c")]
                        for aug, _ in aug_list:
                            try:
                                aug_img  = aug.augment(img)
                                aug_feat = _extract_feat_from_image(pipe, aug_img)
                                if aug_feat is not None:
                                    X.append(aug_feat)
                                    y.append(identity)
                                    aug_saved += 1
                            except Exception:
                                pass
                continue   # skip full extraction below

            # ── Cache miss: load image and extract fresh ──────────────────────
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            # ── Original image ────────────────────────────────────────────────
            feat = _extract_feat_from_image(pipe, img)
            if feat is not None:
                X.append(feat)
                y.append(identity)
                saved += 1
                # Save to cache for next retrain
                save_cached_feature(img_path, active_modalities, feat)

                # ── Augmented variants ────────────────────────────────────────
                if use_augment:
                    aug_list = [
                        (aug_mild,     "mild"),
                        (aug_moderate, "moderate"),
                        (aug_severe,   "severe"),
                    ]
                    if few_images:
                        aug_list += [(aug_mild_b, "mild_b"), (aug_mild_c, "mild_c")]
                    for aug, _ in aug_list:
                        try:
                            aug_img  = aug.augment(img)
                            aug_feat = _extract_feat_from_image(pipe, aug_img)
                            if aug_feat is not None:
                                X.append(aug_feat)
                                y.append(identity)
                                aug_saved += 1
                        except Exception:
                            pass
            else:
                skipped += 1

        extra_note  = " (extra aug: small gallery)" if few_images and use_augment else ""
        aug_info    = f" + {aug_saved} augmented" if use_augment else ""
        cache_note  = f" [{cache_hits} from cache]" if cache_hits else ""
        _log(job_id, f"  {identity}: {saved} original{aug_info}{cache_note}{extra_note}, {skipped} skipped.")

    _log(job_id, f"Total: {len(X)} feature vectors from {len(set(y))} identities "
                 f"({'with adaptive augmentation' if use_augment else 'without augmentation'}).")

    return np.array(X, dtype=np.float32), y


def _run_retrain(job_id: str, use_grid_search: bool = False) -> None:
    """
    Background worker: re-train PCA + SVM from gallery, save to disk,
    then hot-swap the in-memory pipeline singleton.

    Parameters
    ----------
    use_grid_search : bool
        If True, run GridSearchCV to find optimal SVM C and gamma.
        Takes ~5 minutes but can improve accuracy by 5-15%.
    """
    with _jobs_lock:
        _jobs[job_id]["status"]          = "running"
        _jobs[job_id]["started"]         = datetime.utcnow().isoformat()
        _jobs[job_id]["use_grid_search"] = use_grid_search

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
        if use_grid_search:
            _log(job_id, "GridSearchCV enabled - searching best C/gamma (this may take ~5 min)...")
            pipe.classifier.use_grid_search = True
        else:
            pipe.classifier.use_grid_search = False
        pipe.train(X, y)
        gs_note = " (GridSearchCV)" if use_grid_search else ""
        _log(job_id, f"SVM classifier trained successfully{gs_note}.")

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
                        # Use pre-crop fast path for small images.
                        # Video frame thumbnails are saved at max 200px — raise
                        # threshold to 220 so they use the direct-resize path
                        # instead of face detection (which fails on tiny faces).
                        h, w = img.shape[:2]
                        if max(h, w) <= 220:
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
def start_retrain(use_grid_search: bool = False):
    """
    Kick off a background retrain from the current gallery.
    Returns immediately with a job_id. Poll /retrain/status/{job_id} for updates.

    Parameters
    ----------
    use_grid_search : bool (query param, default False)
        Set to true to enable GridSearchCV for SVM hyperparameter tuning.
        Recommended after finalising your gallery. Takes ~5 extra minutes.
        Example: POST /api/pipeline/retrain?use_grid_search=true
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
    t = threading.Thread(
        target=_run_retrain,
        args=(job_id,),
        kwargs={"use_grid_search": use_grid_search},
        daemon=True,
    )
    t.start()

    suffix = " (GridSearchCV enabled - will take longer)" if use_grid_search else ""
    return {
        "message":         f"Retrain started{suffix} in the background.",
        "job_id":          job_id,
        "already_running": False,
        "use_grid_search": use_grid_search,
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
