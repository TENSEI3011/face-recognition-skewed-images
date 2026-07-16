"""
job_manager.py
--------------
Simple in-memory background job tracker for long-running experiment runs.
Jobs are identified by a UUID and hold status, stdout, and return code.
"""

import uuid
import threading
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# job_id → job_info dict
_jobs: Dict[str, dict] = {}
_lock = threading.Lock()


def create_job(name: str) -> str:
    """Create a new job entry and return its ID."""
    job_id = str(uuid.uuid4())[:8]
    with _lock:
        _jobs[job_id] = {
            "id":        job_id,
            "name":      name,
            "status":    "pending",   # pending | running | done | error
            "started":   None,
            "finished":  None,
            "output":    "",
            "returncode": None,
        }
    return job_id


def run_job(job_id: str, script_path: str) -> None:
    """
    Launch a Python script as a subprocess and stream its output into the job record.
    Intended to be called in a background thread.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"]  = "running"
        job["started"] = datetime.utcnow().isoformat()

    # Project root = three levels up from this file (web/backend/services/)
    project_root = str(Path(__file__).resolve().parents[3])

    # Extra env: force UTF-8 so Unicode chars in script output don't crash on Windows
    import os as _os
    env = _os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"

    # Windows: CREATE_NO_WINDOW avoids console popup and encoding conflicts
    extra_kwargs = {}
    if sys.platform == "win32":
        extra_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path],  # -u = unbuffered output
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=project_root,
            env=env,
            **extra_kwargs,
        )
        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            with _lock:
                _jobs[job_id]["output"] = "".join(output_lines[-200:])  # keep last 200 lines
        proc.wait()
        with _lock:
            _jobs[job_id]["status"]     = "done" if proc.returncode == 0 else "error"
            _jobs[job_id]["returncode"] = proc.returncode
            _jobs[job_id]["finished"]   = datetime.utcnow().isoformat()

    except Exception as e:
        with _lock:
            _jobs[job_id]["status"]   = "error"
            _jobs[job_id]["output"]  += f"\nException: {e}"
            _jobs[job_id]["finished"] = datetime.utcnow().isoformat()


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list:
    with _lock:
        return list(_jobs.values())


def start_job_thread(job_id: str, script_path: str) -> None:
    """Launch run_job in a daemon thread (non-blocking)."""
    t = threading.Thread(target=run_job, args=(job_id, script_path), daemon=True)
    t.start()
