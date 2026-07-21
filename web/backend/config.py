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
DEFAULT_PCA_VARIANCE = 0.99   # raised from 0.95 — more components → richer SVM subspace (+2-5% accuracy)
DEFAULT_SVM_KERNEL   = "rbf"
DEFAULT_THRESHOLD    = 0.45   # Cosine similarity gate for SVM fallback path
FAISS_THRESHOLD      = 0.38   # Slightly raised from 0.35 — TTA embeddings are more stable
                               #   so we can afford a tighter gate without losing recall.
                               #   Raise to 0.45 if you get false positives, lower to 0.30
                               #   if enrolled persons are not being recognised (tiny faces).
DEFAULT_TOP_K        = 3

# ── Anti-Spoofing / Liveness Detection ────────────────────────────────────────
# Passive liveness check runs BEFORE identity matching.
# Score 0.0 (certain spoof) → 1.0 (certain real). Faces below LIVENESS_THRESHOLD
# are rejected as presentation attacks (printed photo, screen, mask, video replay).
LIVENESS_ENABLED     = True    # Set False to bypass (useful for testing/demo)
LIVENESS_THRESHOLD   = 0.45    # Lowered from 0.50 — passive is now 6-signal fused;
                                # active blink challenge is the hard gate for live webcam.

# ── Active Liveness — Blink Challenge ─────────────────────────────────────────
# Challenge-response: user must blink within BLINK_TIMEOUT_SEC seconds.
# A static photo / video cannot blink on command → defeats screen/print attacks.
BLINK_ENABLED        = True    # Set False to skip blink challenge (testing only)
BLINK_EAR_THRESHOLD  = 0.25   # Eye Aspect Ratio below this → eye closed (blink)
BLINK_TIMEOUT_SEC    = 7.0    # Seconds user has to complete the blink challenge
BLINK_REQUIRED_COUNT = 1      # Number of blinks required to pass
BLINK_CONSEC_FRAMES  = 2      # Min consecutive frames below EAR threshold = blink

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
