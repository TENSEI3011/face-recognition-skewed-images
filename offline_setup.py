"""
offline_setup.py
----------------
One-time setup for FULLY OFFLINE operation.

Downloads everything needed so the system runs with NO internet access:
  1. Google Fonts (Barlow Condensed, IBM Plex Mono, Inter) → web/frontend/fonts/
  2. InsightFace buffalo_l model (ArcFace + SCRFD detector, ~500 MB)
  3. dlib 68-point landmark model (~100 MB)

Run ONCE on a machine with internet, then copy the whole project to
the air-gapped / defence system.

Usage:
    python offline_setup.py           # Download everything
    python offline_setup.py --fonts   # Fonts only
    python offline_setup.py --models  # Models only (no fonts)

Face Recognition on Skewed UAV Images
"""

import os, sys, bz2, io, urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent

# ── Google Fonts woff2 files ──────────────────────────────────────────────────
FONT_FILES = {
    "BarlowCondensed-Medium.woff2":
        "https://fonts.gstatic.com/s/barlowcondensed/v12/HTxwL3I-JCGChYJ8VI-L6OO_au7B43LT3w.woff2",
    "BarlowCondensed-SemiBold.woff2":
        "https://fonts.gstatic.com/s/barlowcondensed/v12/HTxxL3I-JCGChYJ8VI-L6OO_au7B6xTT3g.woff2",
    "BarlowCondensed-Bold.woff2":
        "https://fonts.gstatic.com/s/barlowcondensed/v12/HTxxL3I-JCGChYJ8VI-L6OO_au7B43PT3g.woff2",
    "IBMPlexMono-Regular.woff2":
        "https://fonts.gstatic.com/s/ibmplexmono/v19/-F63fjptAgt5VM-kVkqdyU8n1ioa1Xdm.woff2",
    "IBMPlexMono-Medium.woff2":
        "https://fonts.gstatic.com/s/ibmplexmono/v19/-F6qfjptAgt5VM-kVkqdyU8n3vAOwlBFgg.woff2",
    "IBMPlexMono-SemiBold.woff2":
        "https://fonts.gstatic.com/s/ibmplexmono/v19/-F6qfjptAgt5VM-kVkqdyU8n3oQJwlBFgg.woff2",
    "Inter-Regular.woff2":
        "https://fonts.gstatic.com/s/inter/v19/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfAZ9hiA.woff2",
    "Inter-Medium.woff2":
        "https://fonts.gstatic.com/s/inter/v19/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuI6fAZ9hiA.woff2",
    "Inter-SemiBold.woff2":
        "https://fonts.gstatic.com/s/inter/v19/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuGKYAZ9hiA.woff2",
}

FONTS_DIR    = ROOT / "web" / "frontend" / "fonts"
DLIB_URL     = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
DLIB_BZ2     = ROOT / "models" / "shape_predictor_68_face_landmarks.dat.bz2"
DLIB_DAT     = ROOT / "models" / "shape_predictor_68_face_landmarks.dat"


def _progress(label):
    def _cb(blk, bsz, tot):
        pct = min(blk * bsz * 100 / tot, 100) if tot > 0 else 0
        bar = chr(9608) * int(pct // 2) + chr(9617) * (50 - int(pct // 2))
        print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)
    return _cb


def _dl(url, dest, label):
    try:
        urllib.request.urlretrieve(url, dest, _progress(label))
        print(); return True
    except Exception as e:
        print(f"\n  ERROR: {e}"); return False


def download_fonts():
    print("\n" + "="*60)
    print("  Step 1: Google Fonts → web/frontend/fonts/")
    print("="*60)
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for fname, url in FONT_FILES.items():
        dest = FONTS_DIR / fname
        if dest.exists():
            print(f"  [SKIP] {fname}"); ok += 1; continue
        print(f"  Downloading {fname} ...")
        if _dl(url, dest, fname):
            print(f"  [ OK ] {fname}"); ok += 1
    print(f"\n  {ok}/{len(FONT_FILES)} font files ready.")
    return ok == len(FONT_FILES)


def download_insightface():
    print("\n" + "="*60)
    print("  Step 2: InsightFace buffalo_l (~500 MB)")
    print("="*60)
    cache = Path.home() / ".insightface" / "models" / "buffalo_l"
    if cache.exists() and any(cache.glob("*.onnx")):
        print(f"  [SKIP] Already cached at {cache}"); return True
    print("  Loading InsightFace — downloading buffalo_l ...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           allowed_modules=["detection","recognition"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640,640))
        print(f"  [ OK ] buffalo_l cached at {cache}"); return True
    except ImportError:
        print("  [FAIL] insightface not installed — pip install insightface onnxruntime"); return False
    except Exception as e:
        print(f"  [FAIL] {e}"); return False


def download_dlib():
    print("\n" + "="*60)
    print("  Step 3: dlib 68-point landmark model (~100 MB)")
    print("="*60)
    if DLIB_DAT.exists():
        print(f"  [SKIP] Already at {DLIB_DAT}"); return True
    DLIB_DAT.parent.mkdir(parents=True, exist_ok=True)
    if not _dl(DLIB_URL, DLIB_BZ2, "shape_predictor_68_face_landmarks.dat.bz2"):
        return False
    print("  Extracting ...")
    try:
        with bz2.open(DLIB_BZ2,"rb") as fi, open(DLIB_DAT,"wb") as fo:
            fo.write(fi.read())
        DLIB_BZ2.unlink()
        print(f"  [ OK ] Extracted to {DLIB_DAT}"); return True
    except Exception as e:
        print(f"  [FAIL] {e}"); return False


def download_sr():
    """Pre-download the EDSR x2 Super-Resolution model (~5 MB)."""
    print("\n" + "="*60)
    print("  Step 4: EDSR x2 Super-Resolution model (~5 MB)")
    print("="*60)
    sr_dir   = ROOT / "models" / "sr"
    sr_model = sr_dir / "EDSR_x2.pb"
    if sr_model.exists():
        print(f"  [SKIP] Already at {sr_model}"); return True
    sr_dir.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/nicehash/EDSR_OpenCV/raw/main/EDSR_x2.pb"
    print("  Downloading EDSR_x2.pb ...")
    ok = _dl(url, sr_model, "EDSR_x2.pb")
    if ok:
        print(f"  [ OK ] SR model saved to {sr_model}")
    else:
        print("  [FAIL] SR model download failed (non-critical — SR feature will be disabled).")
    return ok


def write_instructions():
    txt = """# Offline Deployment Guide

## After running offline_setup.py the system works with NO internet.

## Local MongoDB setup (replaces Atlas):
1. Install MongoDB Community: https://www.mongodb.com/try/download/community
2. Start: mongod --dbpath C:\\data\\db
3. Set in .env:
   MONGO_URI=mongodb://localhost:27017
   MONGO_DB_NAME=facerecog_db

## Run without any MongoDB (local disk fallback):
   Delete or comment out MONGO_URI in .env
   (gallery stored as files, no audit log / watchlist)

## Start server (fully offline):
   web\\start.bat
   or: python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000
"""
    p = ROOT / "OFFLINE_DEPLOYMENT.md"
    p.write_text(txt, encoding="utf-8")
    print(f"\n  [ OK ] Guide saved: OFFLINE_DEPLOYMENT.md")


if __name__ == "__main__":
    args = sys.argv[1:]
    fonts_only  = "--fonts"  in args
    models_only = "--models" in args
    do_all = not fonts_only and not models_only

    print("="*60)
    print("  UAV Face Recognition — Offline Asset Setup")
    print("="*60)

    results = {}
    if do_all or fonts_only:  results["fonts"]       = download_fonts()
    if do_all or models_only: results["insightface"] = download_insightface()
    if do_all or models_only: results["dlib"]        = download_dlib()
    if do_all or models_only: results["sr_model"]    = download_sr()
    if do_all:                write_instructions()

    print("\n" + "="*60 + "\n  Summary\n" + "="*60)
    for k, v in results.items():
        print(f"  {'[ OK ]' if v else '[FAIL]'} {k}")

    print()
    if all(results.values()):
        print("  System is ready for fully offline use.")
        print("  See OFFLINE_DEPLOYMENT.md for MongoDB local setup.")
    else:
        print("  Some downloads failed — check errors and retry.")
    sys.exit(0 if all(results.values()) else 1)
