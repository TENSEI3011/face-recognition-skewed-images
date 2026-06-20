"""
run_baseline.py
---------------
Main experiment: Full pipeline on gallery/probe dataset.

Usage:
  python experiments/run_baseline.py \
      --gallery data/gallery \
      --probe   data/probe \
      --results results/baseline

Expected directory structure:
  data/gallery/
    identity_001/ img1.jpg img2.jpg ...
    identity_002/ ...
  data/probe/
    identity_001/ img1.jpg ...

Face Recognition on Skewed UAV Images
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FaceRecognitionPipeline
from src.fusion import ALL_MODALITIES
from evaluation.metrics import full_evaluation_report, print_report
from evaluation.visualizer import (
    plot_roc_curves, plot_cmc_curves, plot_score_distribution,
    plot_confusion_matrix, plot_pca_variance
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline face recognition experiment."
    )
    parser.add_argument("--gallery",  default="data/gallery",
                        help="Path to gallery dataset directory")
    parser.add_argument("--probe",    default="data/probe",
                        help="Path to probe dataset directory")
    parser.add_argument("--results",  default="results/baseline",
                        help="Output directory for results and plots")
    parser.add_argument("--modalities", nargs="+", default=ALL_MODALITIES,
                        choices=ALL_MODALITIES,
                        help="Feature modalities to use")
    parser.add_argument("--pca_variance", type=float, default=0.95,
                        help="PCA variance retention threshold (0–1)")
    parser.add_argument("--svm_kernel", default="rbf",
                        choices=["rbf", "linear"],
                        help="SVM kernel type")
    parser.add_argument("--grid_search", action="store_true",
                        help="Run GridSearchCV for SVM hyperparameter tuning")
    parser.add_argument("--max_gallery", type=int, default=None,
                        help="Max images per identity in gallery (for speed)")
    parser.add_argument("--max_probe", type=int, default=None,
                        help="Max images per identity in probe")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Max rank for CMC curve")
    parser.add_argument("--predictor_path",
                        default="models/shape_predictor_68_face_landmarks.dat",
                        help="Path to dlib shape predictor .dat file")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*65}")
    print(f"  UAV Face Recognition — Baseline Experiment")
    print(f"  Face Recognition | {timestamp}")
    print(f"  Modalities : {args.modalities}")
    print(f"  PCA Var    : {args.pca_variance}")
    print(f"  SVM Kernel : {args.svm_kernel}")
    print(f"{'='*65}\n")

    # -------------------------------------------------------------------------
    # 1. Initialize Pipeline
    # -------------------------------------------------------------------------
    pipeline = FaceRecognitionPipeline(
        modalities=args.modalities,
        pca_variance=args.pca_variance,
        svm_kernel=args.svm_kernel,
        predictor_path=args.predictor_path,
        use_grid_search=args.grid_search,
    )

    # -------------------------------------------------------------------------
    # 2. Load Gallery (Training Set)
    # -------------------------------------------------------------------------
    print(f"[1/5] Loading gallery from: {args.gallery}")
    X_gallery, y_gallery = pipeline.load_dataset(
        args.gallery, max_per_identity=args.max_gallery
    )

    if len(X_gallery) == 0:
        print("ERROR: No gallery features extracted. Check dataset structure and face detector.")
        print("Expected structure: gallery/<identity_name>/*.jpg")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Train (PCA + SVM)
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Training PCA + SVM on {len(y_gallery)} gallery samples...")
    pipeline.train(X_gallery, y_gallery)
    pipeline.save(results_dir / "models")

    # Plot PCA variance curve
    plot_pca_variance(
        pipeline.reducer.explained_variance_ratio_,
        target_variance=args.pca_variance,
        save_path=results_dir / f"pca_variance_{timestamp}.png",
    )

    # -------------------------------------------------------------------------
    # 4. Load Probe (Test Set)
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Loading probe from: {args.probe}")
    X_probe_raw, y_probe = pipeline.load_dataset(
        args.probe, max_per_identity=args.max_probe
    )

    if len(X_probe_raw) == 0:
        print("ERROR: No probe features extracted.")
        sys.exit(1)

    # Transform probe features through PCA
    X_probe_pca = pipeline.reducer.transform(X_probe_raw)
    X_gallery_pca = pipeline.reducer.transform(X_gallery)

    # -------------------------------------------------------------------------
    # 5. Predict Top-K for Probe
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Running identification on {len(y_probe)} probe samples...")
    top_labels, top_proba = pipeline.classifier.predict_top_k(
        X_probe_pca, k=args.top_k
    )

    # -------------------------------------------------------------------------
    # 6. Evaluate
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Computing evaluation metrics...")
    results = full_evaluation_report(
        y_true=y_probe,
        y_pred_top_k=top_labels,
        X_probe=X_probe_pca,
        y_probe=y_probe,
        n_genuine=min(500, len(y_probe) * 2),
        n_impostor=min(500, len(y_probe) * 2),
    )

    print_report(results, label=f"Full Pipeline [{', '.join(args.modalities)}]")

    # Save metrics as JSON
    metrics_to_save = {
        k: float(v) if isinstance(v, (float, np.floating)) else
           v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in results.items()
        if k not in ["roc_fpr", "roc_tpr", "cmc_curve"]  # save separately
    }
    metrics_to_save["cmc_curve"]    = results["cmc_curve"].tolist()
    metrics_to_save["modalities"]   = args.modalities
    metrics_to_save["pca_variance"] = args.pca_variance
    metrics_to_save["n_gallery"]    = len(y_gallery)
    metrics_to_save["n_probe"]      = len(y_probe)
    metrics_to_save["n_identities"] = len(set(y_gallery))
    metrics_to_save["timestamp"]    = timestamp

    with open(results_dir / f"metrics_{timestamp}.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"[Done] Metrics saved → {results_dir}/metrics_{timestamp}.json")

    # -------------------------------------------------------------------------
    # 7. Generate Plots
    # -------------------------------------------------------------------------
    from evaluation.metrics import generate_verification_pairs

    scores, bin_labels, _ = generate_verification_pairs(
        X_probe_pca, y_probe,
        n_genuine=min(500, len(y_probe)),
        n_impostor=min(500, len(y_probe)),
    )

    plot_roc_curves(
        {"Full Pipeline": (results["roc_fpr"], results["roc_tpr"], results["auc"])},
        title="ROC Curve — Baseline Experiment",
        save_path=results_dir / f"roc_{timestamp}.png",
    )

    plot_cmc_curves(
        {"Full Pipeline": results["cmc_curve"]},
        title="CMC Curve — Baseline Experiment",
        save_path=results_dir / f"cmc_{timestamp}.png",
    )

    plot_score_distribution(
        genuine_scores=scores[bin_labels == 1],
        impostor_scores=scores[bin_labels == 0],
        save_path=results_dir / f"score_dist_{timestamp}.png",
    )

    y_pred_rank1 = top_labels[:, 0]
    plot_confusion_matrix(
        y_probe, y_pred_rank1,
        title="Confusion Matrix — Baseline",
        save_path=results_dir / f"confusion_{timestamp}.png",
    )

    print(f"\n[Complete] All results saved to: {results_dir}/")
    print(f"  Rank-1: {results['rank_1']*100:.2f}% | "
          f"EER: {results['eer']*100:.2f}% | "
          f"AUC: {results['auc']:.4f} | "
          f"d': {results['d_prime']:.4f}")


if __name__ == "__main__":
    main()

