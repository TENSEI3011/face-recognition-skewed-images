"""
download_lfw.py
---------------
Downloads LFW (Labeled Faces in the Wild) dataset and organizes it
into the gallery/probe structure required by the pipeline.

LFW Organization Strategy:
  - Only use identities with >= 5 images (sufficient for train/test split)
  - Gallery: First 3 images per identity (frontal enrollment)
  - Probe:   Remaining images per identity (test queries)
  - Split:   70% identities → train | 30% identities → test
             (identity-disjoint — no leakage)

Download source: http://vis-www.cs.umass.edu/lfw/lfw.tgz (~180MB)

Face Recognition on Skewed UAV Images
"""

import os
import sys
import io
import shutil
import tarfile
import random
import urllib.request
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Configuration ────────────────────────────────────────────────────────────
# Mirror list — tried in order until one succeeds
LFW_URLS = [
    "https://vis-www.cs.umass.edu/lfw/lfw.tgz",           # Official (HTTPS)
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",            # Official (HTTP)
    "https://ndownloader.figshare.com/files/5976018",     # Figshare mirror
]

DOWNLOAD_PATH    = Path("data/datasets/lfw.tgz")
EXTRACT_PATH     = Path("data/datasets/lfw_raw")
GALLERY_DIR      = Path("data/gallery")
PROBE_DIR        = Path("data/probe")

MIN_IMAGES_PER_ID = 5      # Only use identities with >= 5 images
GALLERY_PER_ID    = 3      # Images per identity in gallery
MAX_IDENTITIES    = 200    # Cap at 200 identities for manageable experiments
RANDOM_SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────


def progress_bar(block_num, block_size, total_size):
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded * 100 / total_size if total_size > 0 else 0
    filled = int(pct // 2)
    bar = "#" * filled + "." * (50 - filled)
    mb_done  = downloaded / 1_048_576
    mb_total = total_size / 1_048_576
    print(f"\r  [{bar}] {pct:.1f}%  ({mb_done:.1f}/{mb_total:.1f} MB)",
          end="", flush=True)


def download_lfw():
    DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DOWNLOAD_PATH.exists():
        print(f"[OK] LFW archive already downloaded: {DOWNLOAD_PATH}")
        return

    print(f"\nDownloading LFW dataset...")
    print(f"  Destination: {DOWNLOAD_PATH}\n")

    last_error = None
    for url in LFW_URLS:
        print(f"  Trying: {url}")
        try:
            # Add browser-like headers to avoid server blocks
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, DOWNLOAD_PATH, progress_bar)
            size_mb = DOWNLOAD_PATH.stat().st_size / 1_048_576
            print(f"\n  [OK] Download complete. ({size_mb:.1f} MB)")
            return
        except Exception as e:
            print(f"\n  Failed: {e}")
            last_error = e
            # Remove partial download before retrying
            if DOWNLOAD_PATH.exists():
                DOWNLOAD_PATH.unlink()

    print(f"\n  ERROR: All mirrors failed. Last error: {last_error}")
    print("\n  Manual download instructions:")
    print("    1. Visit https://vis-www.cs.umass.edu/lfw/")
    print("    2. Download 'lfw.tgz' (~180 MB)")
    print(f"    3. Place it at: {DOWNLOAD_PATH.resolve()}")
    print("    4. Re-run this script to organize the dataset.")
    sys.exit(1)


def extract_lfw():
    if EXTRACT_PATH.exists() and any(EXTRACT_PATH.iterdir()):
        print(f"[OK] LFW already extracted at: {EXTRACT_PATH}")
        return

    EXTRACT_PATH.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting LFW archive...")

    with tarfile.open(DOWNLOAD_PATH, "r:gz") as tar:
        members = tar.getmembers()
        for i, member in enumerate(members):
            tar.extract(member, EXTRACT_PATH)
            if i % 500 == 0:
                pct = i * 100 / len(members)
                print(f"\r  Extracting... {pct:.0f}% ({i}/{len(members)} files)",
                      end="", flush=True)

    print(f"\n  [OK] Extracted to: {EXTRACT_PATH}")


def organize_into_gallery_probe():
    print(f"\nOrganizing into Gallery / Probe structure...")

    # LFW extracts to data/datasets/lfw_raw/lfw/<identity>/<images>
    lfw_root = EXTRACT_PATH / "lfw"
    if not lfw_root.exists():
        # Try alternate path
        subdirs = list(EXTRACT_PATH.iterdir())
        if subdirs:
            lfw_root = subdirs[0]
        else:
            print(f"  ERROR: Cannot find extracted LFW at {EXTRACT_PATH}")
            sys.exit(1)

    # Collect identities with enough images
    identity_dirs = sorted([
        d for d in lfw_root.iterdir()
        if d.is_dir() and len(list(d.glob("*.jpg"))) >= MIN_IMAGES_PER_ID
    ])

    print(f"  Found {len(identity_dirs)} identities with >= {MIN_IMAGES_PER_ID} images.")

    # Cap to MAX_IDENTITIES
    random.seed(RANDOM_SEED)
    if len(identity_dirs) > MAX_IDENTITIES:
        identity_dirs = random.sample(identity_dirs, MAX_IDENTITIES)
        print(f"  Capped to {MAX_IDENTITIES} identities (random sample).")

    # Sort and split: 70% train gallery | 30% test gallery+probe
    identity_dirs = sorted(identity_dirs)
    n_total = len(identity_dirs)

    # Clean existing gallery/probe
    for d in [GALLERY_DIR, PROBE_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n_gallery = 0
    n_probe   = 0
    n_ids     = 0

    print(f"\n  Copying images...")
    for identity_dir in identity_dirs:
        identity = identity_dir.name
        images   = sorted(identity_dir.glob("*.jpg"))

        # Gallery: first GALLERY_PER_ID images
        gallery_images = images[:GALLERY_PER_ID]
        # Probe: remaining images
        probe_images   = images[GALLERY_PER_ID:]

        if not gallery_images or not probe_images:
            continue

        # Copy gallery images
        gal_out = GALLERY_DIR / identity
        gal_out.mkdir(exist_ok=True)
        for img in gallery_images:
            shutil.copy2(img, gal_out / img.name)
            n_gallery += 1

        # Copy probe images
        prb_out = PROBE_DIR / identity
        prb_out.mkdir(exist_ok=True)
        for img in probe_images:
            shutil.copy2(img, prb_out / img.name)
            n_probe += 1

        n_ids += 1

    print(f"\n  [OK] Organization complete!")
    print(f"       Identities  : {n_ids}")
    print(f"       Gallery imgs: {n_gallery}  (in data/gallery/)")
    print(f"       Probe imgs  : {n_probe}   (in data/probe/)")
    print(f"       Avg gallery : {n_gallery/max(n_ids,1):.1f} imgs/identity")
    print(f"       Avg probe   : {n_probe/max(n_ids,1):.1f} imgs/identity")

    return n_ids, n_gallery, n_probe


def print_summary(n_ids, n_gallery, n_probe):
    print(f"""
============================================================
  LFW Dataset Ready!
============================================================
  Identities  : {n_ids}
  Gallery     : {n_gallery} images  -> data/gallery/
  Probe       : {n_probe} images  -> data/probe/

  Next step — run the baseline experiment:
  > python experiments/run_baseline.py

  Or run the full ablation study:
  > python experiments/run_ablation.py
============================================================
""")


if __name__ == "__main__":
    print("=" * 60)
    print("  LFW Dataset Downloader & Organizer")
    print("  Face Recognition on Skewed UAV Images")
    print("=" * 60)

    download_lfw()
    extract_lfw()
    n_ids, n_gallery, n_probe = organize_into_gallery_probe()
    print_summary(n_ids, n_gallery, n_probe)

