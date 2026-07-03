"""
create_custom_dataset.py
------------------------
Creates a face recognition dataset from 360-degree face videos.

Usage
-----
1. Record a slow 360-degree face video for EACH person (mp4/avi/mov).
2. Place videos in:  data/videos/<person_name>.mp4
   e.g.  data/videos/alice.mp4
         data/videos/bob.mp4
3. Run:  python create_custom_dataset.py

Output
------
  data/gallery/<person_name>/  ← 3 best frontal frames (enrollment)
  data/probe/<person_name>/    ← remaining diverse frames (query/test)

Tips for recording
------------------
  - Slow rotation: 1 full turn in 10-15 seconds
  - Distance: 0.5-1m from camera
  - Also tilt head slightly up/down (simulate UAV top-down angle)
  - Try different lighting conditions for robustness
  - Resolution: at least 720p

Face Recognition on Skewed UAV Images
"""

import os
import sys
import io
import cv2
import shutil
import argparse
import numpy as np
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Configuration ─────────────────────────────────────────────────────────────
VIDEO_DIR        = Path("data/videos")          # Put your .mp4 files here
GALLERY_DIR      = Path("data/gallery")
PROBE_DIR        = Path("data/probe")

FRAME_SAMPLE_RATE    = 3        # Extract every Nth frame (3 = 10 frames/sec at 30fps)
FACE_SIZE            = (112, 112)
GALLERY_PER_ID       = 3        # Frames saved to gallery per person
MIN_FACE_CONFIDENCE  = 0.85     # MTCNN minimum confidence
BLUR_THRESHOLD       = 4.0      # Laplacian variance on full frame — below = too blurry
SIMILARITY_THRESHOLD = 0.92     # Skip frames too similar to previous kept frame
EYE_ASPECT_RATIO_MIN = 0.15     # Min eye aspect ratio — below = eyes closed (skip)
# ──────────────────────────────────────────────────────────────────────────────


def load_detector():
    """Load MTCNN face detector."""
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        print("[OK] MTCNN detector loaded.")
        return detector
    except ImportError:
        print("[ERROR] MTCNN not installed. Run: pip install mtcnn")
        sys.exit(1)


def is_blurry(image: np.ndarray, threshold: float = None) -> bool:
    """Return True if image is too blurry (Laplacian variance below threshold)."""
    if threshold is None:
        threshold = BLUR_THRESHOLD  # read global at call time, not definition time
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def has_eyes_open(detection: dict, min_ear: float = None) -> bool:
    """
    Return True if the detected face has eyes open, using MTCNN keypoints.
    Uses the vertical span of each eye relative to face height as a proxy
    for Eye Aspect Ratio (EAR). Skips frames where eyes appear closed.
    """
    if min_ear is None:
        min_ear = EYE_ASPECT_RATIO_MIN
    kp = detection.get('keypoints', {})
    if not kp:
        return True  # No keypoints available — don't filter
    box = detection.get('box', [0, 0, 1, 1])
    face_h = max(box[3], 1)  # face bounding-box height

    left_eye  = kp.get('left_eye')
    right_eye = kp.get('right_eye')
    left_mouth  = kp.get('mouth_left')
    right_mouth = kp.get('mouth_right')

    if not (left_eye and right_eye and left_mouth and right_mouth):
        return True  # Missing keypoints — don't filter

    # Eye-to-mouth vertical distance (proxy for eye openness relative to face)
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2
    eye_y   = (left_eye[1]  + right_eye[1])  / 2
    eye_to_mouth = mouth_y - eye_y

    # Inter-eye horizontal distance (proxy for face width seen)
    eye_width = abs(right_eye[0] - left_eye[0])
    if eye_width == 0:
        return True

    # Closed eyes pull eye_y DOWN (closer to mouth), reducing ratio
    ratio = eye_to_mouth / max(eye_width, 1)
    return ratio > min_ear


def is_too_similar(face: np.ndarray, kept_faces: list, threshold: float = None) -> bool:
    """Return True if this face is too similar to any already kept frame."""
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD  # read global at call time, not definition time
    if not kept_faces:
        return False
    gray_new = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
    norm_new = np.linalg.norm(gray_new)
    if norm_new == 0:
        return True
    for kept in kept_faces[-5:]:  # Compare against last 5 kept frames only
        gray_kept = cv2.cvtColor(kept, cv2.COLOR_BGR2GRAY).flatten().astype(np.float32)
        norm_kept = np.linalg.norm(gray_kept)
        if norm_kept == 0:
            continue
        similarity = np.dot(gray_new, gray_kept) / (norm_new * norm_kept)
        if similarity > threshold:
            return True
    return False


def detect_and_crop_face(image: np.ndarray, detector, max_dim: int = 640) -> np.ndarray | None:
    """
    Detect the largest face in the image using MTCNN.
    Returns a 112x112 cropped face or None if not found.

    Downscales large frames to max_dim before detection to avoid OOM on
    high-resolution videos (e.g. 4K .MOV files).
    """
    h, w = image.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)   # only shrink, never enlarge

    if scale < 1.0:
        small = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        small = image

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    if not detections:
        return None

    # Pick highest-confidence detection
    best = max(detections, key=lambda r: r['confidence'])
    if best['confidence'] < MIN_FACE_CONFIDENCE:
        return None

    # Scale bounding box back to original resolution
    x, y, bw, bh = [int(v / scale) for v in best['box']]

    # Add 20% padding around face
    pad_x = int(bw * 0.2)
    pad_y = int(bh * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + bw + pad_x)
    y2 = min(image.shape[0], y + bh + pad_y)

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None

    return cv2.resize(face, FACE_SIZE)



def extract_frames_from_video(
    video_path: Path,
    person_name: str,
    detector,
    sample_rate: int = FRAME_SAMPLE_RATE,
) -> list:
    """
    Extract face frames from a single video file.
    Applies blur detection, eye-open check, and similarity deduplication.
    Returns list of good face crops.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_s   = total_frames / fps if fps > 0 else 0

    print(f"  Video: {video_path.name}")
    print(f"    Duration : {duration_s:.1f}s  |  FPS: {fps:.0f}  |  Frames: {total_frames}")

    good_faces   = []
    frame_idx    = 0
    checked      = 0
    skipped_blur = 0
    skipped_eyes = 0
    skipped_dup  = 0
    no_face      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every Nth frame
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        checked += 1

        # Skip blurry frames
        if is_blurry(frame):
            skipped_blur += 1
            frame_idx += 1
            continue

        # Detect face (get full detection dict with keypoints)
        h, w = frame.shape[:2]
        scale = min(640 / max(h, w), 1.0)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb)

        if not detections:
            no_face += 1
            frame_idx += 1
            continue

        # Pick highest-confidence detection
        best = max(detections, key=lambda r: r['confidence'])
        if best['confidence'] < MIN_FACE_CONFIDENCE:
            no_face += 1
            frame_idx += 1
            continue

        # Skip closed-eye frames
        if not has_eyes_open(best):
            skipped_eyes += 1
            frame_idx += 1
            continue

        # Crop face from original resolution
        x, y, bw, bh = [int(v / scale) for v in best['box']]
        pad_x = int(bw * 0.2)
        pad_y = int(bh * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + bw + pad_x)
        y2 = min(frame.shape[0], y + bh + pad_y)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            no_face += 1
            frame_idx += 1
            continue
        face = cv2.resize(face, FACE_SIZE)

        # Skip if too similar to recently kept frames
        if is_too_similar(face, good_faces):
            skipped_dup += 1
            frame_idx += 1
            continue

        good_faces.append(face)
        frame_idx += 1

    cap.release()

    print(f"    Checked  : {checked} sampled frames")
    print(f"    No face  : {no_face}  |  Blurry: {skipped_blur}  |  Eyes closed: {skipped_eyes}  |  Duplicate: {skipped_dup}")
    print(f"    Kept     : {len(good_faces)} diverse face frames")

    return good_faces


def save_dataset(person_name: str, faces: list, gallery_count: int = GALLERY_PER_ID) -> tuple:
    """
    Split faces into gallery (first N) and probe (rest).
    Saves to data/gallery/<person>/ and data/probe/<person>/.
    Returns (n_gallery, n_probe).
    """
    if len(faces) < gallery_count + 1:
        print(f"  [WARN] Only {len(faces)} frames — need at least {gallery_count + 1}. Skipping.")
        return 0, 0

    gal_dir = GALLERY_DIR / person_name
    prb_dir = PROBE_DIR   / person_name
    gal_dir.mkdir(parents=True, exist_ok=True)
    prb_dir.mkdir(parents=True, exist_ok=True)

    for i, face in enumerate(faces[:gallery_count]):
        cv2.imwrite(str(gal_dir / f"{person_name}_gallery_{i:03d}.jpg"), face)

    for i, face in enumerate(faces[gallery_count:]):
        cv2.imwrite(str(prb_dir / f"{person_name}_probe_{i:03d}.jpg"), face)

    return gallery_count, len(faces) - gallery_count


def print_summary(results: dict):
    total_gallery = sum(v[0] for v in results.values())
    total_probe   = sum(v[1] for v in results.values())
    n_ids         = len([v for v in results.values() if v[0] > 0])

    print(f"""
============================================================
  Custom Dataset Ready!
============================================================
  Identities  : {n_ids}
  Gallery     : {total_gallery} images  -> data/gallery/
  Probe       : {total_probe} images  -> data/probe/

  Per-identity breakdown:
""")
    for name, (ng, np_) in results.items():
        status = "OK" if ng > 0 else "SKIPPED"
        print(f"    [{status}] {name:<20}  gallery={ng}  probe={np_}")

    print("""
  Next step — run the baseline experiment:
  > python experiments/run_baseline.py

  Or run the full ablation study:
  > python experiments/run_ablation.py
============================================================
""")


def main():
    global BLUR_THRESHOLD, SIMILARITY_THRESHOLD  # allow CLI overrides

    parser = argparse.ArgumentParser(
        description="Create face recognition dataset from 360-degree videos."
    )
    parser.add_argument(
        "--video-dir", type=str, default=str(VIDEO_DIR),
        help=f"Directory containing videos (default: {VIDEO_DIR})"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=FRAME_SAMPLE_RATE,
        help=f"Extract every Nth frame (default: {FRAME_SAMPLE_RATE})"
    )
    parser.add_argument(
        "--gallery-count", type=int, default=GALLERY_PER_ID,
        help=f"Frames per identity in gallery (default: {GALLERY_PER_ID})"
    )
    parser.add_argument(
        "--blur-threshold", type=float, default=BLUR_THRESHOLD,
        help=f"Laplacian variance threshold; frames below this are skipped (default: {BLUR_THRESHOLD})"
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for deduplication (default: {SIMILARITY_THRESHOLD})"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Clear existing gallery/probe before processing"
    )
    args = parser.parse_args()

    # Apply CLI overrides
    BLUR_THRESHOLD       = args.blur_threshold
    SIMILARITY_THRESHOLD = args.similarity_threshold

    sample_rate   = args.sample_rate
    gallery_count = args.gallery_count
    video_dir     = Path(args.video_dir)

    print("=" * 60)
    print("  Custom Dataset Creator")
    print("  Face Recognition on Skewed UAV Images")
    print("=" * 60)

    # Check video directory
    if not video_dir.exists():
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[INFO] Created video directory: {video_dir}")
        print(f"       Place your .mp4 videos there, named by person:")
        print(f"       e.g.  {video_dir}/alice.mp4")
        print(f"             {video_dir}/bob.mp4")
        print(f"\n       Then re-run this script.")
        sys.exit(0)

    # Find all video files
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"]
    video_files = [
        f for f in video_dir.iterdir()
        if f.suffix in video_extensions
    ]

    if not video_files:
        print(f"\n[ERROR] No video files found in {video_dir}")
        print(f"        Supported formats: {', '.join(video_extensions)}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(video_files)} video(s):")
    for vf in video_files:
        print(f"  - {vf.name}")

    # Optionally clean existing output
    if args.clean:
        for d in [GALLERY_DIR, PROBE_DIR]:
            if d.exists():
                shutil.rmtree(d)
                print(f"[INFO] Cleared {d}")

    # Load detector once (shared across all videos)
    print("\n[INFO] Loading face detector...")
    detector = load_detector()

    # Process each video
    results = {}
    for video_path in sorted(video_files):
        person_name = video_path.stem  # filename without extension = identity name
        print(f"\n{'─'*60}")
        print(f"  Processing: {person_name}")
        print(f"{'─'*60}")

        faces = extract_frames_from_video(video_path, person_name, detector, sample_rate)

        if faces:
            ng, np_ = save_dataset(person_name, faces, gallery_count)
            results[person_name] = (ng, np_)
            print(f"  [OK] Saved: {ng} gallery + {np_} probe images")
        else:
            results[person_name] = (0, 0)
            print(f"  [WARN] No usable frames extracted for {person_name}")

    print_summary(results)


if __name__ == "__main__":
    main()
