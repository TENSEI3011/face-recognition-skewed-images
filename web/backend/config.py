"""
config.py
---------
Central configuration for the web backend.
All paths are resolved relative to the project root (two levels up from this file).
"""

import os
from pathlib import Path

# ── Project root (Face Recognition/) ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ── Core paths ─────────────────────────────────────────────────────────────────
GALLERY_DIR      = ROOT / "data" / "gallery"
PROBE_DIR        = ROOT / "data" / "probe"
MODELS_DIR       = ROOT / "results" / "baseline" / "models"
RESULTS_DIR      = ROOT / "results"
PREDICTOR_PATH   = ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
FRONTEND_DIR     = Path(__file__).resolve().parents[1] / "frontend"

# ── Pipeline defaults ──────────────────────────────────────────────────────────
DEFAULT_MODALITIES   = ["hog", "lbp", "geometry", "arcface"]
DEFAULT_PCA_VARIANCE = 0.95
DEFAULT_SVM_KERNEL   = "rbf"
DEFAULT_THRESHOLD    = 0.35
DEFAULT_TOP_K        = 3

# ── MongoDB Atlas ──────────────────────────────────────────────────────────────
# Set MONGO_URI as an environment variable on Render / locally in .env
# If not set, the app falls back to local file storage (works on your PC)
MONGO_URI    = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "facerecog_db")

# ── UAV Degradation profiles (maps name → augmentor profile key) ───────────────
DEGRADATION_PROFILES = ["CLEAN", "MILD", "MODERATE", "SEVERE", "EXTREME", "MOTION", "COMBINED"]

# ── Experiment names ───────────────────────────────────────────────────────────
EXPERIMENTS = {
    "baseline":    ROOT / "experiments" / "run_baseline.py",
    "ablation":    ROOT / "experiments" / "run_ablation.py",
    "pose_study":  ROOT / "experiments" / "run_pose_study.py",
    "degradation": ROOT / "experiments" / "run_degradation.py",
}

# Ensure gallery directory exists
GALLERY_DIR.mkdir(parents=True, exist_ok=True)
