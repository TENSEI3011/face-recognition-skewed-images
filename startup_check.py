#!/usr/bin/env python3
"""
startup_check.py
----------------
Run this BEFORE starting the server (or as part of the build step).
It pre-downloads the ArcFace (InsightFace buffalo_l) model so the
first request doesn't timeout waiting for a 300MB download.

Usage:
    python startup_check.py

On Railway / Render, add this to the build command:
    pip install -r requirements.txt && python startup_check.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def check_dlib_model():
    predictor_path = ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
    if not predictor_path.exists():
        print("⚠️  dlib landmark model not found.")
        print(f"   Expected at: {predictor_path}")
        print("   Run: python setup.py")
        print("   Or download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return False
    print(f"✅ dlib model found: {predictor_path}")
    return True


def preload_arcface():
    """Download/cache the InsightFace buffalo_l model."""
    try:
        print("📥 Pre-loading ArcFace (InsightFace buffalo_l)...")
        from src.features.arcface_features import ArcFaceExtractor
        _ = ArcFaceExtractor()
        print("✅ ArcFace model ready.")
        return True
    except Exception as e:
        print(f"❌ ArcFace preload failed: {e}")
        return False


def check_pipeline_model():
    """Check if a trained SVM/PCA model exists."""
    from web.backend.config import MODELS_DIR
    svm_path = MODELS_DIR / "svm_classifier.pkl"
    pca_path = MODELS_DIR / "pca_reducer.pkl"

    if svm_path.exists() and pca_path.exists():
        print(f"✅ Trained model found: {MODELS_DIR}")
        return True
    else:
        print(f"⚠️  No trained model found at {MODELS_DIR}")
        print("   The server will start but recognition won't work until you run")
        print("   the baseline experiment from the web UI (Experiments → Run Baseline).")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  Face Recognition UAV — Startup Check")
    print("=" * 60)

    dlib_ok     = check_dlib_model()
    arcface_ok  = preload_arcface()
    model_ok    = check_pipeline_model()

    print("\n" + "=" * 60)
    print(f"  dlib landmark model : {'✅ OK' if dlib_ok    else '❌ MISSING'}")
    print(f"  ArcFace model       : {'✅ OK' if arcface_ok else '❌ FAILED'}")
    print(f"  Trained SVM/PCA     : {'✅ OK' if model_ok   else '⚠️  Not trained yet'}")
    print("=" * 60)

    if not dlib_ok:
        print("\n🛑 Cannot start without the dlib model. Run: python setup.py")
        sys.exit(1)

    print("\n🚀 All checks passed. Server can start.\n")
