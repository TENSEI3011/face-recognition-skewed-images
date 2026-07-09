"""
retrain.py — Hot-retrain the face recognition pipeline from the gallery.

POST /api/pipeline/retrain
    Starts a background job that:
      1. Loads all gallery images and extracts features
      2. Trains PCA + SVM on the new gallery
      3. Saves the model to disk
      4. Hot-swaps the in-memory pipeline (live stream picks it up instantly)
    Returns a job_id that the frontend can poll.

GET /api/pipeline/retrain/status/{job_id}
    Returns the current status + log output of the retrain job.
"""

import sys
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

# ── Project root on path ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from web.backend.services import pipeline_service
from web.backend.config   import GALLERY_DIR, MODELS_DIR

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# ── In-memory job store ────────────────────────────────────────────────────────
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
    print(line)   # also visible in server console


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

        # ── Validate gallery ───────────────────────────────────────────────────
        if not GALLERY_DIR.exists():
            raise RuntimeError(f"Gallery directory not found: {GALLERY_DIR}")

        identity_dirs = [d for d in GALLERY_DIR.iterdir() if d.is_dir()]
        if len(identity_dirs) < 2:
            raise RuntimeError(
                f"Need at least 2 identities in the gallery to train. "
                f"Found: {len(identity_dirs)}"
            )
        _log(job_id, f"Gallery has {len(identity_dirs)} identities.")

        # ── Get a pipeline instance (reuse loaded models for feature extractors) ─
        # We re-use the existing pipeline's extractors (ArcFace, HOG, etc.) —
        # only PCA + SVM need to be retrained. This avoids reloading 300 MB models.
        pipe = pipeline_service.get_pipeline()
        if pipe is None:
            raise RuntimeError(
                "No pipeline loaded. Run the baseline experiment first to "
                "initialize the pipeline models."
            )

        # ── Extract features from every gallery image ──────────────────────────
        _log(job_id, "Extracting features from gallery images…")
        X, y = pipe.load_dataset(str(GALLERY_DIR), verbose=False)

        if len(X) == 0:
            raise RuntimeError(
                "No faces detected in any gallery image. "
                "Check that images are correctly formatted and faces are visible."
            )

        _log(job_id, f"Extracted {len(X)} feature vectors from {len(set(y))} identities.")

        # ── Train PCA + SVM ────────────────────────────────────────────────────
        _log(job_id, "Fitting PCA reducer…")
        pipe.train(X, y)
        _log(job_id, "SVM classifier trained.")

        # ── Save to disk ───────────────────────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pipe.save(str(MODELS_DIR))
        _log(job_id, f"Model saved to {MODELS_DIR}.")

        # ── Hot-swap the singleton so the live stream picks up immediately ──────
        # pipeline_service already holds the *same* pipe object we just trained
        # (we got it via get_pipeline()), so _is_trained=True and the classifier
        # is already updated in-memory.  We just need to mark the service as loaded.
        pipeline_service._status["loaded"] = True
        pipeline_service._status["error"]  = None

        _log(job_id, "✅ Pipeline hot-swapped. Live stream will use the new model immediately.")

        with _jobs_lock:
            _jobs[job_id]["status"]   = "done"
            _jobs[job_id]["finished"] = datetime.utcnow().isoformat()

    except Exception as exc:
        _log(job_id, f"❌ Error: {exc}")
        with _jobs_lock:
            _jobs[job_id]["status"]   = "error"
            _jobs[job_id]["error"]    = str(exc)
            _jobs[job_id]["finished"] = datetime.utcnow().isoformat()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/retrain")
def start_retrain():
    """
    Kick off a background retrain from the current gallery.
    Returns immediately with a job_id.  Poll /retrain/status/{job_id} for updates.
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
