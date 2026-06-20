"""
run_pose_study.py
-----------------
Pose-stratified evaluation experiment.

Evaluates Rank-1 accuracy separately for each pose bin by
loading probe images organized by pose angle.

Expected probe directory structure:
  data/probe_pose/
    yaw_0/    identity_001/ img1.jpg ...
    yaw_15/   identity_001/ ...
    yaw_30/   ...
    yaw_45/   ...
    yaw_60/   ...
    yaw_90/   ...
    pitch_-30/ ...
    pitch_-45/ ...
    pitch_-60/ ...
    pitch_-90/ ...

Or: use --synthesize flag to generate pose-varied probes from gallery
    using Albumentations + rotation as proxy.

Usage:
  python experiments/run_pose_study.py \
      --gallery data/gallery \
      --probe_pose_dir data/probe_pose \
      --results results/pose_study

Face Recognition on Skewed UAV Images
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FaceRecognitionPipeline
from src.augmentation import UAVAugmentor, DegradationProfile
from evaluation.metrics import full_evaluation_report
from evaluation.visualizer import plot_rank1_vs_pose, plot_cmc_curves


# Pose bins to evaluate
YAW_BINS   = ["yaw_0", "yaw_15", "yaw_30", "yaw_45", "yaw_60", "yaw_90"]
PITCH_BINS = ["pitch_-10", "pitch_-20", "pitch_-30", "pitch_-45", "pitch_-60", "pitch_-90"]
ALTITUDE_BINS = ["alt_5m", "alt_10m", "alt_15m", "alt_20m", "alt_25m", "alt_30m"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pose-stratified face recognition evaluation."
    )
    parser.add_argument("--gallery",         default="data/gallery")
    parser.add_argument("--probe_pose_dir",  default="data/probe_pose",
                        help="Directory with per-pose-bin probe subfolders")
    parser.add_argument("--results",         default="results/pose_study")
    parser.add_argument("--pca_variance",    type=float, default=0.95)
    parser.add_argument("--svm_kernel",      default="rbf")
    parser.add_argument("--max_per_bin",     type=int, default=None)
    parser.add_argument("--predictor_path",
                        default="models/shape_predictor_68_face_landmarks.dat")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_base = Path(args.probe_pose_dir)

    print(f"\n{'='*65}")
    print(f"  UAV Face Recognition — Pose Stratified Study")
    print(f"  Face Recognition | {timestamp}")
    print(f"{'='*65}\n")

    # -------------------------------------------------------------------------
    # 1. Train on gallery (once)
    # -------------------------------------------------------------------------
    print("[1/3] Training pipeline on gallery...")
    pipeline = FaceRecognitionPipeline(
        pca_variance=args.pca_variance,
        svm_kernel=args.svm_kernel,
        predictor_path=args.predictor_path,
    )
    X_gallery, y_gallery = pipeline.load_dataset(args.gallery)
    if len(X_gallery) == 0:
        print("ERROR: Empty gallery. Check data directory.")
        sys.exit(1)
    pipeline.train(X_gallery, y_gallery)

    # -------------------------------------------------------------------------
    # 2. Evaluate per pose bin
    # -------------------------------------------------------------------------
    print("\n[2/3] Evaluating per pose bin...")
    all_pose_results = {}
    all_cmc = {}

    # Try yaw bins, pitch bins, altitude bins in order
    bin_groups = {
        "Yaw":     YAW_BINS,
        "Pitch":   PITCH_BINS,
        "Altitude": ALTITUDE_BINS,
    }

    for group_name, bins in bin_groups.items():
        group_results = {}
        group_cmc = {}
        group_found = False

        for bin_name in bins:
            bin_dir = probe_base / bin_name
            if not bin_dir.exists():
                continue
            group_found = True

            X_probe_raw, y_probe = pipeline.load_dataset(
                bin_dir,
                max_per_identity=args.max_per_bin,
                verbose=False,
            )
            if len(X_probe_raw) == 0:
                print(f"  [{bin_name}] No samples extracted — skipping")
                continue

            X_probe_pca = pipeline.reducer.transform(X_probe_raw)
            top_labels, _ = pipeline.classifier.predict_top_k(X_probe_pca, k=10)

            res = full_evaluation_report(
                y_true=y_probe,
                y_pred_top_k=top_labels,
                X_probe=X_probe_pca,
                y_probe=y_probe,
            )
            group_results[bin_name] = res
            group_cmc[bin_name] = res["cmc_curve"]

            print(f"  [{bin_name}] Rank-1: {res['rank_1']*100:.2f}% | "
                  f"EER: {res['eer']*100:.2f}% | "
                  f"n={len(y_probe)}")

        if group_found:
            all_pose_results[group_name] = group_results
            all_cmc[group_name] = group_cmc

    if not all_pose_results:
        print(f"\nNo pose bin directories found under: {probe_base}")
        print("Expected subdirectories like: yaw_0, yaw_30, pitch_-45, alt_10m, etc.")
        print("If you haven't organized your probe data by pose yet, use the")
        print("augmentation pipeline to synthesize pose-varied probes from gallery.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # 3. Save and plot
    # -------------------------------------------------------------------------
    print("\n[3/3] Saving results and generating plots...")

    for group_name, group_results in all_pose_results.items():
        rank1_per_bin = {k: v["rank_1"] for k, v in group_results.items()}

        # Clean label for filenames
        group_slug = group_name.lower().replace(" ", "_")

        plot_rank1_vs_pose(
            rank1_per_bin,
            title=f"Rank-1 Accuracy vs {group_name} ({group_name} Bins)",
            xlabel=f"{group_name} Bin",
            save_path=results_dir / f"rank1_vs_{group_slug}_{timestamp}.png",
        )

        if all_cmc.get(group_name):
            valid = {k: v for k, v in all_cmc[group_name].items()
                     if isinstance(v, np.ndarray) and len(v) > 0}
            if valid:
                plot_cmc_curves(
                    valid,
                    title=f"CMC Curves — {group_name} Stratified",
                    save_path=results_dir / f"cmc_{group_slug}_{timestamp}.png",
                )

    # Save JSON
    save_data = {}
    for group, group_res in all_pose_results.items():
        save_data[group] = {}
        for bin_name, res in group_res.items():
            save_data[group][bin_name] = {
                k: float(v) if isinstance(v, (float, np.floating)) else
                   v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in res.items()
                if k not in ["roc_fpr", "roc_tpr"]
            }

    with open(results_dir / f"pose_study_{timestamp}.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n[Done] Pose study results saved to: {results_dir}/")


if __name__ == "__main__":
    main()

