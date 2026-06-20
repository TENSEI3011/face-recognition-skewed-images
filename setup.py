"""
setup.py
--------
One-time setup script.
Downloads the dlib 68-point landmark model and creates required directories.

Run once before starting experiments:
    python setup.py

Face Recognition on Skewed UAV Images
"""

import os
import bz2
import sys
import io
import urllib.request
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DIRS_TO_CREATE = [
    "data/gallery",
    "data/probe",
    "data/probe_pose",
    "data/datasets",
    "models",
    "results/baseline",
    "results/ablation",
    "results/pose_study",
    "results/degradation",
    "experiments",
]

DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DLIB_BZ2_PATH  = Path("models/shape_predictor_68_face_landmarks.dat.bz2")
DLIB_DAT_PATH  = Path("models/shape_predictor_68_face_landmarks.dat")


def create_directories():
    print("Creating project directories...")
    for d in DIRS_TO_CREATE:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {d}/")


def download_dlib_model():
    if DLIB_DAT_PATH.exists():
        print(f"\n✓ dlib model already exists at: {DLIB_DAT_PATH}")
        return

    print(f"\nDownloading dlib landmark model...")
    print(f"  URL: {DLIB_MODEL_URL}")
    print(f"  This is ~100 MB. Please wait...\n")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
        bar = "#" * int(pct // 2) + "." * (50 - int(pct // 2))
        print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(DLIB_MODEL_URL, DLIB_BZ2_PATH, progress)
        print("\n  Download complete. Extracting...")

        with bz2.open(DLIB_BZ2_PATH, "rb") as f_in, \
             open(DLIB_DAT_PATH, "wb") as f_out:
            f_out.write(f_in.read())

        DLIB_BZ2_PATH.unlink()  # Remove compressed file
        print(f"  [OK] Extracted to: {DLIB_DAT_PATH}")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        print("  Please download manually from: http://dlib.net/files/")
        print(f"  And place the .dat file at: {DLIB_DAT_PATH}")


def create_sample_dataset_note():
    """Create a README inside data/ explaining structure."""
    note = """# Data Directory

Place your datasets here with the following structure:

## Gallery (Enrollment Images)
```
gallery/
  person_001/
    frontal_01.jpg   (High-quality frontal face)
    frontal_02.jpg
  person_002/
    frontal_01.jpg
```

## Probe (Test Images)  
```
probe/
  person_001/
    uav_or_pose_varied_01.jpg
  person_002/
    uav_or_pose_varied_01.jpg
```

## For Quick Testing
You can use the LFW dataset (http://vis-www.cs.umass.edu/lfw/):
- Use 3 images per identity as gallery
- Use remaining images as probe

## For UAV-Specific Testing
Use DroneSURF dataset: https://iab-rubric.org/index.php/dronesurf
Or CFP dataset: http://cfpw.io/ (for pose variation study)
"""
    Path("data/README.md").write_text(note, encoding="utf-8")
    print("\n  [OK] data/README.md created with dataset structure guide.")


if __name__ == "__main__":
    print("=" * 60)
    print("  UAV Face Recognition — Project Setup")
    print("  Face Recognition on Skewed UAV Images")
    print("=" * 60)

    create_directories()
    download_dlib_model()
    create_sample_dataset_note()

    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Add gallery images to:  data/gallery/<identity>/")
    print("  2. Add probe images to:    data/probe/<identity>/")
    print("  3. Run baseline:  python experiments/run_baseline.py")
    print("  4. Run ablation:  python experiments/run_ablation.py")
    print("\nInsightFace (ArcFace) model will auto-download on first run.")

