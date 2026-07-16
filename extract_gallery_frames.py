"""
extract_gallery_frames.py
--------------------------
Extracts the clearest face frames from a video and saves them as gallery images.
Run this when your gallery photos don't match the video conditions (distance, angle, lighting).

Usage:
    python extract_gallery_frames.py --video <video_path> --name <identity_name>

Example:
    python extract_gallery_frames.py --video "14609104-hd_1920_1080_60fps.mp4" --name "lucky"
"""

import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to the video file")
parser.add_argument("--name",  required=True, help="Identity name (folder will be created in data/gallery/<name>)")
parser.add_argument("--max",   type=int, default=30, help="Maximum frames to save (default: 30)")
parser.add_argument("--conf",  type=float, default=0.3, help="Min face detection confidence (default: 0.3)")
args = parser.parse_args()

VIDEO_PATH  = Path(args.video) if Path(args.video).is_absolute() else ROOT / args.video
IDENTITY    = args.name
OUT_DIR     = ROOT / "data" / "gallery" / IDENTITY
MAX_FRAMES  = args.max
MIN_CONF    = args.conf

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Video:    {VIDEO_PATH}")
print(f"Identity: {IDENTITY}")
print(f"Output:   {OUT_DIR}")
print()

# Load detector
from src.detection import FaceDetector
detector = FaceDetector(min_face_size=10, confidence_threshold=MIN_CONF)

cap   = cv2.VideoCapture(str(VIDEO_PATH))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")
print(f"Scanning for face frames...")

# Collect (frame_idx, frame, face_detection, sharpness_score)
candidates = []

for i in range(total):
    ret, frame = cap.read()
    if not ret:
        break

    # Only check every 3rd frame for speed
    if i % 3 != 0:
        continue

    dets = detector.detect(frame)
    if not dets:
        continue

    best = max(dets, key=lambda d: d["confidence"])
    bx, by, bw, bh = best["box"]

    # Crop the face with padding
    pad = int(max(bw, bh) * 0.3)
    x1 = max(0, bx - pad)
    y1 = max(0, by - pad)
    x2 = min(w, bx + bw + pad)
    y2 = min(h, by + bh + pad)
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    # Compute sharpness of the face crop
    gray       = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    face_area  = bw * bh

    # Score = sharpness * face_area (prefer sharp + large faces)
    score = sharpness * (face_area ** 0.5)
    candidates.append((score, i, frame.copy(), best))

cap.release()

print(f"Found {len(candidates)} frames with faces.")

if not candidates:
    print("No faces found in video!")
    sys.exit(1)

# Sort by score, pick top MAX_FRAMES, space them out temporally
candidates.sort(key=lambda x: -x[0])

# De-duplicate: avoid saving nearly identical frames (within 0.5 sec)
selected = []
used_times = []
for score, frame_idx, frame, det in candidates:
    t = frame_idx / fps
    # Skip if too close in time to an already selected frame
    if any(abs(t - ut) < 0.5 for ut in used_times):
        continue
    selected.append((score, frame_idx, frame, det))
    used_times.append(t)
    if len(selected) >= MAX_FRAMES:
        break

print(f"Saving {len(selected)} best frames to {OUT_DIR}/")
print()

# Count existing files to avoid overwriting
existing = list(OUT_DIR.glob("*.jpg")) + list(OUT_DIR.glob("*.png"))
start_idx = len(existing)

for rank, (score, frame_idx, frame, det) in enumerate(selected):
    bx, by, bw, bh = det["box"]
    t = frame_idx / fps

    # Save the full face crop (with padding)
    pad = int(max(bw, bh) * 0.3)
    fh, fw = frame.shape[:2]
    x1 = max(0, bx - pad)
    y1 = max(0, by - pad)
    x2 = min(fw, bx + bw + pad)
    y2 = min(fh, by + bh + pad)
    crop = frame[y1:y2, x1:x2]

    # Upscale tiny faces to at least 112px for gallery usability
    ch, cw = crop.shape[:2]
    if cw < 112 or ch < 112:
        scale_up = max(112 / cw, 112 / ch)
        crop = cv2.resize(crop, (int(cw * scale_up), int(ch * scale_up)),
                          interpolation=cv2.INTER_CUBIC)

    out_path = OUT_DIR / f"{IDENTITY}_video_{start_idx + rank:03d}.jpg"
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [{rank+1:2d}] Frame {frame_idx:4d} ({t:.2f}s): face={bw}x{bh}px  score={score:.0f}  -> {out_path.name}")

print()
print(f"Done! {len(selected)} gallery images saved to: {OUT_DIR}")
print()
print("Next steps:")
print("  1. Open the web UI")
print("  2. Go to Training / Retrain")
print("  3. Click 'Retrain' to rebuild the FAISS index with the new person")
