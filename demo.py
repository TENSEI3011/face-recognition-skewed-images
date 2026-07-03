"""
demo.py
-------
WAR-MOVIE STYLE Face Recognition Demo
======================================
Processes a video file (or webcam) and annotates each frame with:
  - Green bounding box  - Known person (high confidence)
  - Red bounding box    - UNKNOWN (low confidence / not in gallery)
  - Name + confidence % label on the box

Usage
-----
# Identify faces in siddhant's video:
  python demo.py --video data/videos/siddhant.mp4

# Test with a different/unknown person video:
  python demo.py --video path/to/unknown_person.mp4

# Use live webcam:
  python demo.py --webcam

# Save annotated output video:
  python demo.py --video data/videos/siddhant.mp4 --save

# Change confidence threshold (default 0.50):
  python demo.py --video data/videos/siddhant.mp4 --threshold 0.60

Face Recognition on Skewed UAV Images
"""

import sys
import io
import cv2
import numpy as np
import argparse
import time
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import FaceRecognitionPipeline
from src.reducer import PCAReducer
from src.classifier import SVMClassifier

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR         = Path("results/baseline/models")
PREDICTOR_PATH    = "models/shape_predictor_68_face_landmarks.dat"
DEFAULT_THRESHOLD = 0.35    # Below this confidence -> UNKNOWN
# NOTE: With 3 identities, SVM Platt probabilities range ~33-70%.
# 0.50 was too strict — correct predictions were being labelled UNKNOWN.
DISPLAY_SCALE     = 0.45    # Scale for large display (helps with 4K videos)
MIN_DISPLAY_LONG  = 720     # Minimum long-edge pixels for the output window
                            # Small videos (e.g. Siddhant 464x832) are scaled UP
                            # so the HUD/labels are as readable as Stuti/Aditi.
PROCESS_EVERY_N   = 3       # Process every Nth frame (speed vs accuracy)
MAX_DIM_DETECT    = 640     # Max dim for detection (memory safety)
SMOOTH_WINDOW     = 30      # Temporal smoother: rolling window of N predictions
SMOOTH_MIN_VOTE   = 0.60    # Majority threshold: label must win > this fraction
                            # 0.60 blocks Stuti flicker without triggering on head turns

# ── Face-detection quality gates ─────────────────────────────────────────────
DET_CONFIDENCE_MIN = 0.97   # MTCNN confidence floor — filters non-face detections
FACE_ASPECT_MIN    = 0.55   # min(w/h, h/w) — reject very elongated boxes (clothes/hands)
FACE_MIN_AREA_FRAC = 0.003  # face area must be >= 0.3% of frame area (rejects tiny blobs)
EYE_DIST_MIN_FRAC  = 0.10   # eye-eye distance must be >= 10% of face width

# ── Colors (BGR) ────────────────────────────────────────────────────────────
COLOR_KNOWN   = (0, 220, 80)    # Green  - known person
COLOR_UNKNOWN = (0, 60, 220)    # Red    - unknown person
COLOR_ACCENT  = (255, 200, 0)   # Cyan   - header
COLOR_BG      = (20, 20, 20)    # Dark background for text

IDENTITY_COLORS = [
    (0, 220, 80),
    (255, 150, 0),
    (200, 80, 255),
    (0, 220, 220),
    (80, 255, 200),
]
# ────────────────────────────────────────────────────────────────────────────


class TemporalSmoother:
    """
    Per-face rolling-window majority-vote smoother.

    Keeps the last SMOOTH_WINDOW raw predictions (label, conf) for each
    tracked face slot. On each query it returns the label that appears most
    often — if that label's vote share is above SMOOTH_MIN_VOTE it is
    returned as the stable prediction; otherwise 'UNKNOWN' is returned.

    This eliminates brief 1-2 frame misidentifications caused by hard
    viewing angles without altering the underlying model.
    """

    def __init__(self, window: int = SMOOTH_WINDOW, min_vote: float = SMOOTH_MIN_VOTE):
        self.window   = window
        self.min_vote = min_vote
        # One deque per face slot (we only track one face in this demo)
        from collections import deque
        self._buffers: dict[int, deque] = {}

    def update(self, slot: int, label: str, conf: float) -> tuple[str, float]:
        """
        Push a new raw prediction and return the smoothed (label, avg_conf).
        """
        from collections import deque
        if slot not in self._buffers:
            self._buffers[slot] = deque(maxlen=self.window)
        buf = self._buffers[slot]
        buf.append((label, conf))

        # Count votes
        from collections import Counter
        counts = Counter(lbl for lbl, _ in buf)
        top_label, top_count = counts.most_common(1)[0]
        vote_share = top_count / len(buf)

        if vote_share >= self.min_vote:
            # Average confidence for the winning label only
            avg_conf = sum(c for lbl, c in buf if lbl == top_label) / top_count
            return top_label, avg_conf
        else:
            return "UNKNOWN", 0.0

    def query(self, slot: int) -> tuple[str, float]:
        """
        Return the current smoothed verdict WITHOUT updating the buffer.
        Used for uncertain frames (conf < threshold) so they don't pollute
        the majority vote — the last stable result is held instead.
        """
        from collections import Counter
        buf = self._buffers.get(slot)
        if not buf:
            return "UNKNOWN", 0.0

        counts = Counter(lbl for lbl, _ in buf)
        top_label, top_count = counts.most_common(1)[0]
        vote_share = top_count / len(buf)

        if vote_share >= self.min_vote:
            avg_conf = sum(c for lbl, c in buf if lbl == top_label) / top_count
            return top_label, avg_conf
        return "UNKNOWN", 0.0

    def reset(self, slot: int = None) -> None:
        """Clear buffer(s) — call between videos."""
        if slot is None:
            self._buffers.clear()
        else:
            self._buffers.pop(slot, None)


def load_pipeline(model_dir: Path, threshold: float):
    pca_path = model_dir / "pca_reducer.pkl"
    svm_path = model_dir / "svm_classifier.pkl"

    if not pca_path.exists() or not svm_path.exists():
        print(f"\n[ERROR] Trained models not found in: {model_dir}")
        print("        Run first:  python experiments/run_baseline.py")
        sys.exit(1)

    # ── Auto-detect modalities from saved metadata ────────────────────────────
    modalities = FaceRecognitionPipeline.load_modalities(model_dir)
    print(f"[Demo] Detected modalities from saved model: {modalities}")

    print(f"[Demo] Loading trained models from: {model_dir}")
    pipeline = FaceRecognitionPipeline(
        modalities=modalities,          # must match what was used during training!
        predictor_path=PREDICTOR_PATH,
    )
    pipeline.reducer     = PCAReducer.load(pca_path)
    pipeline.classifier  = SVMClassifier.load(svm_path)
    pipeline._is_trained = True

    identities = list(pipeline.classifier.classes_)
    print(f"[Demo] Enrolled identities ({len(identities)}): {identities}")
    print(f"[Demo] UNKNOWN threshold: {threshold:.0%}")

    color_map = {
        name: IDENTITY_COLORS[i % len(IDENTITY_COLORS)]
        for i, name in enumerate(sorted(identities))
    }
    return pipeline, identities, color_map


def _is_valid_face(det: dict, frame_h: int, frame_w: int) -> bool:
    """
    Quality-gate for MTCNN detections.
    Rejects non-face regions (clothing, background) that MTCNN occasionally fires on.

    Checks:
      1. High confidence (>= DET_CONFIDENCE_MIN)
      2. Aspect ratio is roughly square (no very tall/wide boxes)
      3. Minimum face area relative to frame
      4. Eye-keypoints are present and spaced properly (real face geometry)
    """
    conf = det.get('confidence', 0)
    if conf < DET_CONFIDENCE_MIN:
        return False

    x, y, bw, bh = det['box']
    if bw <= 0 or bh <= 0:
        return False

    # Aspect ratio check — face box should be roughly square
    aspect = min(bw / bh, bh / bw)
    if aspect < FACE_ASPECT_MIN:
        return False

    # Minimum area check — filter tiny/partial detections
    face_area = bw * bh
    frame_area = frame_h * frame_w
    if face_area < FACE_MIN_AREA_FRAC * frame_area:
        return False

    # Keypoint geometry check — eyes must be present and horizontally separated
    kp = det.get('keypoints', {})
    left_eye  = kp.get('left_eye')
    right_eye = kp.get('right_eye')
    if left_eye and right_eye:
        eye_dist = abs(right_eye[0] - left_eye[0])
        if eye_dist < EYE_DIST_MIN_FRAC * bw:
            return False   # Eyes too close — not a real forward-facing face

    return True


def detect_faces_scaled(frame, detector):
    h, w = frame.shape[:2]
    scale = min(MAX_DIM_DETECT / max(h, w), 1.0)
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

    raw_detections = detector.detect(small)

    # Scale boxes back to original frame coordinates first
    if scale < 1.0:
        for d in raw_detections:
            d['box'] = [int(v / scale) for v in d['box']]
            if 'keypoints' in d:
                d['keypoints'] = {
                    k: (int(pt[0] / scale), int(pt[1] / scale))
                    for k, pt in d['keypoints'].items()
                }

    # Apply face-quality gates (filters clothing/background false positives)
    valid = [d for d in raw_detections if _is_valid_face(d, h, w)]
    return valid


def draw_label_box(frame, x, y, w, h, label, confidence, color):
    """Draw stylish bounding box with corner accents and name label."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Corner accents
    clen = max(10, min(w, h) // 5)
    tk   = 3
    for cx, cy, dx, dy in [(x, y, 1, 1), (x+w, y, -1, 1), (x, y+h, 1, -1), (x+w, y+h, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * clen, cy), color, tk)
        cv2.line(frame, (cx, cy), (cx, cy + dy * clen), color, tk)

    # Label text
    font       = cv2.FONT_HERSHEY_DUPLEX
    fscale     = max(0.45, min(0.9, w / 200))
    name_text  = label.upper()
    conf_text  = f"{confidence:.0%}"

    (lw, lh), _ = cv2.getTextSize(name_text, font, fscale, 1)
    (cw, ch), _ = cv2.getTextSize(conf_text, font, fscale * 0.8, 1)

    bw      = max(lw, cw) + 16
    bh      = lh + ch + 18
    label_y = max(y - bh - 4, 0)

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, label_y), (x + bw, label_y + bh), COLOR_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Color stripe on left
    cv2.rectangle(frame, (x, label_y), (x + 4, label_y + bh), color, cv2.FILLED)

    # Text
    cv2.putText(frame, name_text, (x + 10, label_y + lh + 4),
                font, fscale, color, 1, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (x + 10, label_y + lh + ch + 12),
                font, fscale * 0.8, (200, 200, 200), 1, cv2.LINE_AA)


def draw_hud(frame, frame_idx, fps_actual, identities, color_map, threshold):
    """Draw top HUD bar and bottom legend."""
    h, w = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    cv2.putText(frame, "UAV FACE RECOGNITION SYSTEM",
                (12, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Frame: {frame_idx:05d}   FPS: {fps_actual:.1f}   Threshold: {threshold:.0%}",
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # Bottom legend
    lx, ly = 12, h - 16
    for name in sorted(identities):
        col = color_map[name]
        cv2.circle(frame, (lx + 6, ly - 4), 6, col, cv2.FILLED)
        cv2.putText(frame, name.capitalize(), (lx + 18, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        lx += len(name) * 10 + 36

    # Unknown in legend
    cv2.circle(frame, (lx + 6, ly - 4), 6, COLOR_UNKNOWN, cv2.FILLED)
    cv2.putText(frame, "UNKNOWN", (lx + 18, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_UNKNOWN, 1, cv2.LINE_AA)


def process_video(source, pipeline, color_map, identities, threshold, save_path=None, show=True,
                  smooth_window=SMOOTH_WINDOW, smooth_min_vote=SMOOTH_MIN_VOTE):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    ow     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    src_label = "Webcam" if source == 0 else Path(str(source)).name
    print(f"\n[Demo] Source : {src_label}")
    print(f"[Demo] Size   : {ow}x{oh}  FPS: {fps_in:.0f}  Frames: {total}")

    # Adaptive scale: shrink 4K, but ensure minimum size for small videos
    long_edge = max(ow, oh)
    scale = max(DISPLAY_SCALE, MIN_DISPLAY_LONG / long_edge)
    dw = int(ow * scale)
    dh = int(oh * scale)

    writer = None
    if save_path:
        # XVID in AVI container — natively playable on Windows without codec packs
        # save_path should have .avi extension (set in main())
        out_fps = min(fps_in, 60.0)  # cap at 60 fps for safety
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(str(save_path), fourcc, out_fps, (dw, dh))
        if writer.isOpened():
            print(f"[Demo] Codec   : XVID (AVI)")
            print(f"[Demo] Saving to: {save_path}")
        else:
            writer.release()
            writer = None
            print("[Demo] WARNING: Could not open VideoWriter with XVID codec!")

    print("[Demo] Controls: Q = quit | SPACE = pause | S = snapshot")
    print("[Demo] Running...\n")

    frame_idx    = 0
    last_results = []
    paused       = False
    t_start      = time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    smoother     = TemporalSmoother(window=smooth_window, min_vote=smooth_min_vote)  # majority-vote rolling window

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

        fps_actual = frame_idx / max(time.time() - t_start, 1e-6)
        disp       = cv2.resize(frame, (dw, dh))

        # Detect + identify every N frames; only keep the LARGEST face.
        # Multiple MTCNN detections can include false-positive regions
        # (background/clothing) whose ArcFace embeddings match wrong identities.
        # In a single-subject video the largest box is always the real person.
        if not paused and (frame_idx % PROCESS_EVERY_N == 1 or frame_idx == 1):
            raw_dets     = detect_faces_scaled(frame, pipeline.detector)
            last_results = []

            if raw_dets:
                det  = max(raw_dets, key=lambda d: d['box'][2] * d['box'][3])
                x, y, w, h = det['box']
                x, y = max(0, x), max(0, y)
                face_crop = pipeline.detector.crop_face(frame, [x, y, w, h])
                if face_crop is not None and face_crop.size > 0:
                    try:
                        result = pipeline.identify(face_crop, top_k=1)
                        if result is not None:
                            labels, confs = result
                            raw_name, raw_conf = labels[0], confs[0]

                            # Apply temporal majority-vote smoothing.
                            # Only push to smoother when confidence is above threshold
                            # — uncertain frames skip the update so the stable majority
                            # is not diluted by ambiguous predictions (head-turn frames).
                            if raw_conf >= threshold:
                                smooth_name, smooth_conf = smoother.update(0, raw_name, raw_conf)
                            else:
                                # Conf too low: don't update smoother, use its last verdict
                                smooth_name, smooth_conf = smoother.query(0)

                            is_known = smooth_name != "UNKNOWN"
                            last_results.append({
                                'box':      [int(v * scale) for v in [x, y, w, h]],
                                'name':     smooth_name,
                                'conf':     smooth_conf if is_known else raw_conf,
                                'is_known': is_known,
                            })
                    except Exception as e:
                        print(f"[Demo] Identification error (frame {frame_idx}): {e}", file=sys.stderr)

        # Draw all results
        for res in last_results:
            x, y, w, h = res['box']
            color = color_map.get(res['name'], COLOR_UNKNOWN) if res['is_known'] else COLOR_UNKNOWN
            draw_label_box(disp, x, y, w, h, res['name'], res['conf'], color)

        draw_hud(disp, frame_idx, fps_actual, identities, color_map, threshold)

        if writer:
            writer.write(disp)
            if frame_idx % 100 == 0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"[Demo] Progress: frame {frame_idx}/{total_frames} ({pct:.1f}%)")

        if show:
            try:
                cv2.imshow("UAV Face Recognition | Q to quit", disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('s'):
                    snap = Path("results") / f"snapshot_{frame_idx:05d}.jpg"
                    snap.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(snap), disp)
                    print(f"[Demo] Snapshot -> {snap}")
                elif key == ord(' '):
                    paused = not paused
                    print(f"[Demo] {'PAUSED' if paused else 'RESUMED'}")
            except cv2.error:
                # opencv-python-headless installed - no GUI support, skip display
                show = False
                print("[Demo] Note: GUI not available (headless OpenCV). Saving video only.")

    cap.release()
    if writer:
        writer.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless OpenCV - no windows to destroy
    print(f"\n[Demo] Done. Processed {frame_idx} frames.")
    if save_path:
        print(f"[Demo] Saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="UAV Face Recognition Live Demo")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video",  type=str, help="Path to input video file")
    src.add_argument("--webcam", action="store_true", help="Use live webcam")

    parser.add_argument("--model-dir",  default=str(MODEL_DIR))
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help="Min confidence to call known (default 0.35)")
    parser.add_argument("--save",       action="store_true",
                        help="Save annotated output video to results/demo_output/")
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress display window (headless mode)")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW,
                        help=f"Temporal smoother rolling window size (default {SMOOTH_WINDOW})")
    parser.add_argument("--smooth-min-vote", type=float, default=SMOOTH_MIN_VOTE,
                        help=f"Temporal smoother majority threshold (default {SMOOTH_MIN_VOTE})")
    args = parser.parse_args()

    print("=" * 60)
    print("  UAV Face Recognition - Live Demo")
    print("  Face Recognition on Skewed UAV Images")
    print("=" * 60)

    if args.webcam:
        source, src_name = 0, "webcam"
    elif args.video:
        source, src_name = args.video, Path(args.video).stem
    else:
        source, src_name = "data/videos/siddhant.mp4", "siddhant"
        print(f"[Demo] No source given - defaulting to: {source}")
        print(f"[Demo] Tip: python demo.py --video data/videos/aditi.MOV\n")

    pipeline, identities, color_map = load_pipeline(Path(args.model_dir), args.threshold)

    save_path = None
    if args.save:
        out_dir = Path("results/demo_output")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"demo_{src_name}.avi"  # AVI+XVID plays natively on Windows

    process_video(
        source         = source,
        pipeline       = pipeline,
        color_map      = color_map,
        identities     = identities,
        threshold      = args.threshold,
        save_path      = save_path,
        show           = not args.no_display,
        smooth_window  = args.smooth_window,
        smooth_min_vote= args.smooth_min_vote,
    )


if __name__ == "__main__":
    main()
