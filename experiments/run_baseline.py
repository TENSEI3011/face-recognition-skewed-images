"""
run_baseline.py
---------------
WHAT : Full pipeline baseline evaluation \u2014 trains on the gallery (enrollment)
       and evaluates on the probe (query) set with all four feature modalities.
WHY  : The baseline establishes the upper-bound performance of the system
       under ideal conditions (gallery = good quality, probe = same or similar
       conditions). All other experiments (ablation, degradation, pose) compare
       their results against this baseline.

       TRAIN/TEST SPLIT DISCIPLINE:
         Gallery (data/gallery/) = enrollment images per identity (3\u201320 photos each)
         Probe   (data/probe/)   = query images per identity (distinct from gallery)
         WHY SEPARATE DIRS instead of random split: In real deployments, gallery
         images are taken in controlled conditions (passport photo, registration);
         probe images come from surveillance cameras. Using separate directories
         mimics this realistic scenario rather than random 80/20 splits.

       OUTPUTS:
         - Rank-1, Rank-5, Rank-10 CMC accuracy
         - EER, TAR@FAR=0.1%, AUC (verification metrics)
         - d-prime discriminability
         - ROC, CMC, confusion matrix plots
         - Excel results spreadsheet

Usage:
  python experiments/run_baseline.py \\
      --gallery data/gallery \\
      --probe   data/probe \\
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

# Fix Windows cp1252 encoding — must be before any print() that uses Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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


def _get_identity_dirs(root: str) -> set:
    """Return the set of identity names (subdirectory names) in a dataset root."""
    p = Path(root)
    if not p.exists():
        return set()
    return {d.name for d in p.iterdir() if d.is_dir()}


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
    # 0. Gallery / Probe intersection check  ← NEW
    # -------------------------------------------------------------------------
    # WHY THIS MATTERS:
    #   The baseline trains the SVM on gallery identities and evaluates on probe.
    #   If gallery has identities NOT in probe (e.g. a newly enrolled person with
    #   no probe images yet), the SVM learns an extra class that has no probe
    #   images to match against. This "phantom" class steals confident predictions
    #   from the real probe identities, causing accuracy to collapse (88% → 28%).
    #
    #   Fix: restrict both gallery and probe to the INTERSECTION of identities.
    #   Gallery-only identities are still enrolled for FAISS/web-UI identification
    #   but excluded from baseline evaluation until probe images are added.
    gallery_ids = _get_identity_dirs(args.gallery)
    probe_ids   = _get_identity_dirs(args.probe)

    gallery_only = gallery_ids - probe_ids   # enrolled but no probe images yet
    probe_only   = probe_ids   - gallery_ids  # probe but not enrolled
    common_ids   = gallery_ids & probe_ids

    print(f"[0/5] Identity set check:")
    print(f"      Gallery  : {sorted(gallery_ids)}")
    print(f"      Probe    : {sorted(probe_ids)}")
    print(f"      Common   : {sorted(common_ids)}  ← will be evaluated")

    if gallery_only:
        print(f"\n  ⚠  Gallery-only (no probe images — EXCLUDED from baseline):")
        for name in sorted(gallery_only):
            print(f"       • {name}  →  add images to data/probe/{name}/ to include")
        print(f"     These identities are still recognised by the web UI (FAISS).")

    if probe_only:
        print(f"\n  ⚠  Probe-only (not enrolled — EXCLUDED from baseline):")
        for name in sorted(probe_only):
            print(f"       • {name}  →  add images to data/gallery/{name}/ to include")

    if len(common_ids) < 2:
        print("\nERROR: Need at least 2 identities in BOTH gallery and probe to run baseline.")
        print("Add probe images for your enrolled identities to data/probe/<name>/ and retry.")
        sys.exit(1)

    # Build per-identity filter string used by load_dataset
    # We pass allowed_identities so the loader skips excluded dirs.
    allowed = common_ids
    print(f"\n  Proceeding with {len(allowed)} identities: {sorted(allowed)}\n")

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
    # 2. Load Gallery (Training Set) — only common identities
    # -------------------------------------------------------------------------
    print(f"[1/5] Loading gallery from: {args.gallery}")
    X_gallery_all, y_gallery_all = pipeline.load_dataset(
        args.gallery, max_per_identity=args.max_gallery
    )

    # Filter to common identities only
    mask_g = [lbl in allowed for lbl in y_gallery_all]
    X_gallery = np.array([X_gallery_all[i] for i, m in enumerate(mask_g) if m])
    y_gallery  = [y_gallery_all[i] for i, m in enumerate(mask_g) if m]

    if len(X_gallery) == 0:
        print("ERROR: No gallery features extracted. Check dataset structure and face detector.")
        print("Expected structure: gallery/<identity_name>/*.jpg")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Train (PCA + SVM) — only on common identities
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Training PCA + SVM on {len(y_gallery)} gallery samples "
          f"({len(set(y_gallery))} identities)...")
    pipeline.train(X_gallery, y_gallery)
    pipeline.save(results_dir / "models")

    # Plot PCA variance curve
    plot_pca_variance(
        pipeline.reducer.explained_variance_ratio_,
        target_variance=args.pca_variance,
        save_path=results_dir / f"pca_variance_{timestamp}.png",
    )

    # -------------------------------------------------------------------------
    # 4. Load Probe (Test Set) — auto-balanced, common identities only
    # -------------------------------------------------------------------------
    # WHY AUTO-BALANCE:
    #   If one identity has many more probe images than others (e.g. siddhant=64
    #   vs aditi=9), that identity dominates the Rank-1 accuracy calculation.
    #   A single identity at 68% of probe means even perfect classification of
    #   the other 32% can only bring total accuracy to ~32% if that identity is
    #   systematically hard. Balancing gives a fair view of per-class accuracy.
    print(f"\n[3/5] Loading probe from: {args.probe}")

    # Count available probe images per common identity to compute balance cap
    probe_counts = {}
    for name in allowed:
        id_dir = Path(args.probe) / name
        if id_dir.exists():
            imgs = (
                list(id_dir.glob("*.jpg"))  + list(id_dir.glob("*.jpeg")) +
                list(id_dir.glob("*.png"))  + list(id_dir.glob("*.bmp"))
            )
            probe_counts[name] = len(imgs)
        else:
            probe_counts[name] = 0

    # Cap per identity: use the user-specified max_probe if set;
    # otherwise auto-balance to min(count) capped at 20 to avoid tiny probe sets
    if args.max_probe:
        balance_cap = args.max_probe
        print(f"  Probe cap (user-specified): {balance_cap} images/identity")
    else:
        min_count  = min(probe_counts.values()) if probe_counts else 1
        balance_cap = min(max(min_count, 3), 20)   # at least 3, at most 20
        print(f"  Probe images per identity: " +
              ", ".join(f"{n}={c}" for n, c in sorted(probe_counts.items())))
        print(f"  Auto-balance cap: {balance_cap} images/identity "
              f"(min={min_count}, capped at 20)")

    X_probe_all, y_probe_all = pipeline.load_dataset(
        args.probe, max_per_identity=balance_cap
    )

    # Filter to common identities only
    mask_p = [lbl in allowed for lbl in y_probe_all]
    X_probe_raw = np.array([X_probe_all[i] for i, m in enumerate(mask_p) if m])
    y_probe     = [y_probe_all[i] for i, m in enumerate(mask_p) if m]

    if len(X_probe_raw) == 0:
        print("ERROR: No probe features extracted.")
        sys.exit(1)

    from collections import Counter
    probe_dist = Counter(y_probe)
    print(f"  Final probe distribution: " +
          ", ".join(f"{n}={c}" for n, c in sorted(probe_dist.items())))

    # Transform probe features through PCA
    X_probe_pca   = pipeline.reducer.transform(X_probe_raw)
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
    print(f"[Done] Metrics saved -> {results_dir}/metrics_{timestamp}.json")

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

