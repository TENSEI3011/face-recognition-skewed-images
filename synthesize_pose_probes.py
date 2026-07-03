"""
synthesize_pose_probes.py
--------------------------
Synthesizes pose-stratified probe images from gallery/probe images
for the pose study experiment.

Pose simulation approach:
  - Yaw  : 2D in-plane rotation (proxy for head turning left/right)
  - Pitch : perspective warp (proxy for UAV top-down angle)
  - Altitude : resolution downsampling (UAVAugmentor.altitude_downsample)

Output structure:
  data/probe_pose/
    yaw_0/    <identity>/ img_*.jpg
    yaw_15/   <identity>/ ...
    yaw_30/   <identity>/ ...
    yaw_45/   <identity>/ ...
    yaw_60/   <identity>/ ...
    yaw_90/   <identity>/ ...
    pitch_-10/ <identity>/ ...
    pitch_-20/ <identity>/ ...
    pitch_-30/ <identity>/ ...
    pitch_-45/ <identity>/ ...
    pitch_-60/ <identity>/ ...
    alt_5m/   <identity>/ ...
    alt_10m/  <identity>/ ...
    alt_20m/  <identity>/ ...
    alt_30m/  <identity>/ ...

Usage:
  python synthesize_pose_probes.py
  python synthesize_pose_probes.py --source data/probe --n_per_bin 5

Face Recognition on Skewed UAV Images
"""

import sys
import io
import cv2
import numpy as np
import argparse
import shutil
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from src.augmentation import UAVAugmentor


# ── Config ───────────────────────────────────────────────────────────────────
SOURCE_DIR   = Path("data/probe")           # Source images (probe set)
GALLERY_DIR  = Path("data/gallery")         # Also include gallery for more samples
OUTPUT_DIR   = Path("data/probe_pose")
FACE_SIZE    = (112, 112)

# Pose bins
YAW_ANGLES   = [0, 15, 30, 45, 60, 90]     # degrees of in-plane rotation (proxy for yaw)
PITCH_ANGLES = [-10, -20, -30, -45, -60]   # negative = UAV looking down (top-down perspective)
ALTITUDES    = [5, 10, 20, 30]              # metres

N_PER_BIN    = 5   # max images per identity per bin (to keep it fast)
# ─────────────────────────────────────────────────────────────────────────────


def rotate_image(image: np.ndarray, angle_deg: float, border_value=(128, 128, 128)) -> np.ndarray:
    """Rotate image around center by angle_deg degrees (in-plane). Simulates yaw."""
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_value)


def pitch_warp(image: np.ndarray, pitch_deg: float) -> np.ndarray:
    """
    Simulate top-down UAV pitch by applying a perspective warp.

    pitch_deg: negative = UAV looking down from above.
    As |pitch| increases, the image is compressed vertically (top-down view effect).

    The warp compresses the top of the face more than the bottom,
    simulating an aerial viewpoint.
    """
    h, w = image.shape[:2]
    # How much to compress the top of the image (0 = no warp, 1 = fully flat)
    # pitch_deg ranges from 0 (frontal) to -90 (straight down)
    t = min(abs(pitch_deg) / 90.0, 0.95)  # 0..1

    # Top edge compression: top-width narrows, center stays, bottom expands slightly
    compress = t * 0.45  # max horizontal compression at top

    src = np.float32([
        [0,            0],             # top-left
        [w,            0],             # top-right
        [w,            h],             # bottom-right
        [0,            h],             # bottom-left
    ])
    dst = np.float32([
        [w * compress, h * t * 0.1],   # top-left  → moved right & down
        [w * (1 - compress), h * t * 0.1],  # top-right → moved left & down
        [w,            h],             # bottom-right stays
        [0,            h],             # bottom-left stays
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(128, 128, 128))
    return warped


def altitude_sim(image: np.ndarray, altitude_m: float) -> np.ndarray:
    """Simulate UAV altitude by downsampling and upsampling (resolution loss)."""
    return UAVAugmentor.altitude_downsample(image, altitude_m)


def collect_source_images(source_dirs: list[Path], n_per_identity: int = None) -> dict[str, list[np.ndarray]]:
    """
    Collect face images from source directories organized as:
      source_dir/<identity>/*.jpg

    Returns dict: {identity_name: [bgr_image, ...]}
    """
    all_images = {}
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"  [WARN] Source dir not found: {src_dir} — skipping")
            continue

        for id_dir in sorted(src_dir.iterdir()):
            if not id_dir.is_dir():
                continue
            identity = id_dir.name
            imgs = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                for img_path in sorted(id_dir.glob(ext)):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        imgs.append(img)

            if imgs:
                if identity not in all_images:
                    all_images[identity] = []
                all_images[identity].extend(imgs)
                print(f"  {identity}: +{len(imgs)} images from {src_dir.name}")

    # Cap per identity
    if n_per_identity:
        for k in all_images:
            all_images[k] = all_images[k][:n_per_identity]

    return all_images


def save_images(images_by_id: dict[str, list[np.ndarray]],
                bin_dir: Path, bin_label: str,
                transform_fn=None, transform_label="") -> int:
    """Apply transform and save images to bin_dir/<identity>/."""
    total_saved = 0
    for identity, imgs in images_by_id.items():
        out_id_dir = bin_dir / identity
        out_id_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(imgs):
            if transform_fn is not None:
                out = transform_fn(img)
            else:
                out = img.copy()

            # Resize to standard face size
            out = cv2.resize(out, FACE_SIZE)
            fname = f"{identity}_{bin_label}_{i:04d}.jpg"
            cv2.imwrite(str(out_id_dir / fname), out)
            total_saved += 1

    return total_saved


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize pose-stratified probe images for the pose study."
    )
    parser.add_argument("--source", nargs="+",
                        default=[str(SOURCE_DIR), str(GALLERY_DIR)],
                        help="Source image directories (probe/gallery)")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="Output directory for pose-binned probes")
    parser.add_argument("--n_per_bin", type=int, default=N_PER_BIN,
                        help="Max images per identity per bin (default 5)")
    parser.add_argument("--clean", action="store_true",
                        help="Clear output directory before synthesizing")
    args = parser.parse_args()

    output_dir = Path(args.output)
    source_dirs = [Path(s) for s in args.source]

    print("=" * 60)
    print("  Pose Probe Synthesizer")
    print("  Face Recognition on Skewed UAV Images")
    print("=" * 60)
    print(f"\nSource dirs : {[str(s) for s in source_dirs]}")
    print(f"Output dir  : {output_dir}")
    print(f"Images/bin  : {args.n_per_bin}")

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[Clean] Removed {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images
    print("\n[1/4] Collecting source images...")
    source_images = collect_source_images(source_dirs, n_per_identity=args.n_per_bin)
    if not source_images:
        print("[ERROR] No images found in source directories.")
        sys.exit(1)

    identities = sorted(source_images.keys())
    print(f"\nIdentities  : {identities}")
    for name, imgs in source_images.items():
        print(f"  {name}: {len(imgs)} images")

    total_saved = 0

    # ── YAW bins ──────────────────────────────────────────────────────────────
    print(f"\n[2/4] Generating yaw bins {YAW_ANGLES}°...")
    for angle in YAW_ANGLES:
        bin_name = f"yaw_{angle}"
        bin_dir  = output_dir / bin_name
        n = save_images(
            source_images, bin_dir, bin_label=f"yaw{angle}",
            transform_fn=lambda img, a=angle: rotate_image(img, a),
        )
        total_saved += n
        print(f"  [{bin_name}] {n} images saved")

    # ── PITCH bins ────────────────────────────────────────────────────────────
    print(f"\n[3/4] Generating pitch bins {PITCH_ANGLES}°...")
    for angle in PITCH_ANGLES:
        bin_name = f"pitch_{angle}"
        bin_dir  = output_dir / bin_name
        n = save_images(
            source_images, bin_dir, bin_label=f"pitch{angle}",
            transform_fn=lambda img, a=angle: pitch_warp(img, a),
        )
        total_saved += n
        print(f"  [{bin_name}] {n} images saved")

    # ── ALTITUDE bins ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Generating altitude bins {ALTITUDES}m...")
    for alt in ALTITUDES:
        bin_name = f"alt_{alt}m"
        bin_dir  = output_dir / bin_name
        n = save_images(
            source_images, bin_dir, bin_label=f"alt{alt}m",
            transform_fn=lambda img, a=alt: altitude_sim(img, a),
        )
        total_saved += n
        print(f"  [{bin_name}] {n} images saved")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pose probes generated!")
    print(f"  Total images : {total_saved}")
    print(f"  Output       : {output_dir}/")
    print(f"\n  Bins created:")
    for b in sorted(output_dir.iterdir()):
        if b.is_dir():
            n = sum(1 for f in b.rglob("*.jpg"))
            print(f"    {b.name}: {n} images")
    print(f"\n  Next step:")
    print(f"  > python experiments/run_pose_study.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
