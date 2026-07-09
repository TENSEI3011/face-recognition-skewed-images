"""
experiments.py — Experiment trigger router
Allows triggering background experiment runs and polling their status.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from web.backend.config import EXPERIMENTS
from web.backend.services import job_manager

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/jobs")
def list_jobs():
    """List all experiment jobs and their status."""
    return {"jobs": job_manager.list_jobs()}


@router.post("/run/{experiment}")
def run_experiment(experiment: str):
    """Trigger a background experiment run."""
    if experiment not in EXPERIMENTS:
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment}")

    script = EXPERIMENTS[experiment]
    if not Path(script).exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script}")

    job_id = job_manager.create_job(name=experiment)
    job_manager.start_job_thread(job_id, str(script))

    return {
        "job_id":     job_id,
        "experiment": experiment,
        "message":    f"Experiment '{experiment}' started. Poll /api/experiments/status/{job_id} for updates.",
    }


@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Poll status of a running or completed job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job
