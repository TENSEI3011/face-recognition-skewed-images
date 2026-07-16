"""
run_degradation.py
------------------
WHAT : Degradation sweep experiment — evaluates recognition accuracy as
       UAV imaging conditions progressively worsen.
WHY  : The primary threat to accuracy in UAV face recognition is degradation
       (blur, noise, compression) caused by altitude, vibration, and transmission.
       This experiment answers: "At what degradation level does accuracy become
       unacceptable, and which modalities degrade fastest?"

       Two sweep modes:
         1. PROFILE SWEEP: Apply each DegradationProfile (CLEAN→COMBINED) to probe
            images before feature extraction. Shows accuracy vs. severity level.
         2. ALTITUDE SWEEP: Apply altitude_downsample() at 5m, 10m, 15m, 20m, 25m, 30m.
            Shows the specific effect of distance-induced resolution loss.

       WHY TRAIN ON CLEAN GALLERY: We always enroll with good-quality images.
       The question is how well the system identifies people from degraded probes.
       Training on degraded data would mask the real-world performance gap.

Usage:
  python experiments/run_degradation.py \
      --gallery data/gallery \
      --probe   data/probe \
      --results results/degradation

Face Recognition on Skewed UAV Images
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Fix Windows cp1252 encoding — must be before any print() that uses Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FaceRecognitionPipeline
from src.augmentation import UAVAugmentor, DegradationProfile
from evaluation.metrics import full_evaluation_report
from evaluation.visualizer import plot_degradation_curve, plot_rank1_vs_pose


# Ordered degradation profiles for sweep
DEGRADATION_SWEEP = [
    DegradationProfile.CLEAN,
    DegradationProfile.MILD,
    DegradationProfile.MODERATE,
    DegradationProfile.SEVERE,
    DegradationProfile.EXTREME,
    DegradationProfile.MOTION,
    DegradationProfile.COMBINED,
]

# Altitude sweep (simulate specific altitudes)
ALTITUDE_SWEEP_M = [5, 10, 15, 20, 25, 30]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Degradation sweep evaluation."
    )
    parser.add_argument("--gallery",     default="data/gallery")
    parser.add_argument("--probe",       default="data/probe")
    parser.add_argument("--results",     default="results/degradation")
    parser.add_argument("--pca_variance", type=float, default=0.95)
    parser.add_argument("--svm_kernel",  default="rbf")
    parser.add_argument("--max_probe",   type=int, default=None)
    parser.add_argument("--predictor_path",
                        default="models/shape_predictor_68_face_landmarks.dat")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*65}")
    print(f"  UAV Face Recognition — Degradation Sweep")
    print(f"  Face Recognition | {timestamp}")
    print(f"{'='*65}\n")

    # -------------------------------------------------------------------------
    # Train pipeline (clean gallery, no augmentation)
    # -------------------------------------------------------------------------
    print("[1/3] Training pipeline on clean gallery...")
    pipeline = FaceRecognitionPipeline(
        pca_variance=args.pca_variance,
        svm_kernel=args.svm_kernel,
        predictor_path=args.predictor_path,
    )
    X_gallery, y_gallery = pipeline.load_dataset(args.gallery)
    if len(X_gallery) == 0:
        print("ERROR: Empty gallery.")
        sys.exit(1)
    pipeline.train(X_gallery, y_gallery)

    # -------------------------------------------------------------------------
    # Degradation Profile Sweep
    # -------------------------------------------------------------------------
    print("\n[2/3] Running degradation profile sweep...")
    profile_results = {}

    for profile in DEGRADATION_SWEEP:
        name = profile.value
        aug = UAVAugmentor(profile=profile)
        print(f"\n  Profile: [{name.upper()}]")

        X_probe_raw, y_probe = pipeline.load_dataset(
            args.probe,
            augmentor=aug,
            max_per_identity=args.max_probe,
            verbose=False,
        )
        if len(X_probe_raw) == 0:
            print(f"  -> No samples. Skipping.")
            continue

        X_probe_pca = pipeline.reducer.transform(X_probe_raw)
        top_labels, _ = pipeline.classifier.predict_top_k(X_probe_pca, k=10)

        res = full_evaluation_report(
            y_true=y_probe,
            y_pred_top_k=top_labels,
            X_probe=X_probe_pca,
            y_probe=y_probe,
        )
        profile_results[name] = res
        print(f"  -> Rank-1: {res['rank_1']*100:.2f}% | "
              f"EER: {res['eer']*100:.2f}% | "
              f"AUC: {res['auc']:.4f}")

    # -------------------------------------------------------------------------
    # Altitude Simulation Sweep
    # -------------------------------------------------------------------------
    print("\n[3/3] Running altitude simulation sweep...")
    altitude_results = {}

    for altitude_m in ALTITUDE_SWEEP_M:
        label = f"{altitude_m}m"
        print(f"\n  Altitude: {label}")

        X_probe_raw, y_probe = pipeline.load_dataset(
            args.probe,
            augmentor=None,
            max_per_identity=args.max_probe,
            verbose=False,
        )
        if len(X_probe_raw) == 0:
            continue

        # Apply altitude downsampling post-load
        # (reload with altitude applied directly to images before feature extraction)
        pipeline2 = FaceRecognitionPipeline(
            pca_variance=args.pca_variance,
            svm_kernel=args.svm_kernel,
            predictor_path=args.predictor_path,
        )
        pipeline2.reducer = pipeline.reducer
        pipeline2.classifier = pipeline.classifier
        pipeline2._is_trained = True

        import cv2
        from pathlib import Path as P
        probe_dir = P(args.probe)
        X_alt, y_alt = [], []
        for id_dir in sorted(probe_dir.iterdir()):
            if not id_dir.is_dir():
                continue
            identity = id_dir.name
            for img_path in list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.png")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = UAVAugmentor.altitude_downsample(img, altitude_m)
                feat = pipeline.extract_features(img)
                if feat is not None:
                    X_alt.append(feat)
                    y_alt.append(identity)

        if len(X_alt) == 0:
            print(f"  -> No features extracted.")
            continue

        X_alt = np.array(X_alt)
        X_alt_pca = pipeline.reducer.transform(X_alt)
        top_labels, _ = pipeline.classifier.predict_top_k(X_alt_pca, k=10)

        res = full_evaluation_report(
            y_true=y_alt,
            y_pred_top_k=top_labels,
            X_probe=X_alt_pca,
            y_probe=y_alt,
        )
        altitude_results[label] = res
        print(f"  -> Rank-1: {res['rank_1']*100:.2f}% | n={len(y_alt)}")

    # -------------------------------------------------------------------------
    # Save and Plot
    # -------------------------------------------------------------------------
    if profile_results:
        plot_degradation_curve(
            profile_results,
            x_label="Degradation Profile",
            title="Rank-1 Accuracy vs Degradation Severity",
            save_path=results_dir / f"degradation_sweep_{timestamp}.png",
        )

    if altitude_results:
        plot_rank1_vs_pose(
            {k: v["rank_1"] for k, v in altitude_results.items()},
            title="Rank-1 Accuracy vs UAV Altitude",
            xlabel="Altitude (m)",
            save_path=results_dir / f"altitude_sweep_{timestamp}.png",
        )

    # Save JSON
    def safe_val(v):
        if isinstance(v, (float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    save_data = {
        "profile_sweep": {
            k: {kk: safe_val(vv) for kk, vv in v.items()
                if kk not in ["roc_fpr", "roc_tpr"]}
            for k, v in profile_results.items()
        },
        "altitude_sweep": {
            k: {kk: safe_val(vv) for kk, vv in v.items()
                if kk not in ["roc_fpr", "roc_tpr"]}
            for k, v in altitude_results.items()
        },
        "timestamp": timestamp,
    }
    with open(results_dir / f"degradation_{timestamp}.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n[Done] Degradation study results saved to: {results_dir}/")


if __name__ == "__main__":
    main()

