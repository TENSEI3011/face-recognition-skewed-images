"""
run_ablation.py
---------------
WHAT : Ablation study — systematically tests all 10 feature modality combinations
       to quantify how much each modality contributes to recognition accuracy.
WHY  : An ablation study is essential to justify design choices in a research paper.
       Without it, we cannot know whether ArcFace alone is sufficient or whether
       HOG/LBP/Geometry add meaningful value. Results typically show:
         - ArcFace alone is strong at frontal faces
         - HOG+LBP help at extreme pose where ArcFace degrades
         - Full pipeline (all 4) achieves best average performance

       WHY 10 COMBINATIONS: There are 2^4 - 1 = 15 non-empty subsets of 4 modalities,
       but we choose the 10 most meaningful ones (single modalities, useful pairs,
       and progressively adding modalities toward the full pipeline).
       This covers the key questions:
         "Does adding geometry help?" (HOG+LBP vs HOG+LBP+Geometry)
         "Is ArcFace doing most of the work?" (ArcFace only vs All)

Usage:
  python experiments/run_ablation.py \
      --gallery data/gallery \
      --probe   data/probe \
      --results results/ablation

Face Recognition on Skewed UAV Images"""

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
from src.fusion import ALL_MODALITIES
from evaluation.metrics import full_evaluation_report, print_report
from evaluation.visualizer import plot_ablation_study, plot_cmc_curves


# ── Ablation Configurations ───────────────────────────────────────────────────
# Each entry: display_name → list of modality strings passed to FaceRecognitionPipeline
# Ordered to build up from single modality baselines to the full pipeline
ABLATION_CONFIGS = {
    # Single-modality baselines — show the raw contribution of each feature type
    "HOG only":                   ["hog"],
    "LBP only":                   ["lbp"],
    "Geometry only":              ["geometry"],
    "ArcFace only":               ["arcface"],
    # Two-modality combinations — test classical vs deep fusion
    "HOG + LBP":                  ["hog", "lbp"],       # Classical fusion
    "HOG + ArcFace":              ["hog", "arcface"],   # Gradient + deep
    "LBP + ArcFace":              ["lbp", "arcface"],   # Texture + deep
    # Three-modality
    "HOG + LBP + Geometry":       ["hog", "lbp", "geometry"],   # Classical only (no deep)
    "HOG + LBP + ArcFace":        ["hog", "lbp", "arcface"],    # Classical + deep, no geometry
    # Full pipeline — should achieve best overall performance
    "All (Full Pipeline)":        ["hog", "lbp", "geometry", "arcface"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation study over feature modalities."
    )
    parser.add_argument("--gallery",         default="data/gallery")
    parser.add_argument("--probe",           default="data/probe")
    parser.add_argument("--results",         default="results/ablation")
    parser.add_argument("--pca_variance",    type=float, default=0.95)
    parser.add_argument("--svm_kernel",      default="rbf")
    parser.add_argument("--max_gallery",     type=int, default=None)
    parser.add_argument("--max_probe",       type=int, default=None)
    parser.add_argument("--predictor_path",
                        default="models/shape_predictor_68_face_landmarks.dat")
    return parser.parse_args()


def run_single_ablation(
    modalities: list[str],
    gallery_dir: str,
    probe_dir: str,
    args,
) -> dict:
    """
    Train and evaluate the pipeline for a specific modality combination.

    WHY RE-TRAIN FOR EACH COMBINATION: Each modality set produces a different
    fused feature vector dimension (e.g., HOG-only = 3780-D vs. All = 4963-D).
    PCA and SVM must be re-fitted on the correct dimensionality for each run.
    Caching gallery features per modality would save time but adds complexity.

    Returns a results dict with all evaluation metrics, or a failure-safe
    dict with zeroed metrics if data is unavailable.
    """
    pipeline = FaceRecognitionPipeline(
        modalities=modalities,
        pca_variance=args.pca_variance,
        svm_kernel=args.svm_kernel,
        predictor_path=args.predictor_path,
    )

    X_gallery, y_gallery = pipeline.load_dataset(
        gallery_dir, max_per_identity=args.max_gallery, verbose=False
    )
    if len(X_gallery) == 0:
        return {"rank_1": 0, "eer": 1.0, "auc": 0.5, "d_prime": 0.0}

    pipeline.train(X_gallery, y_gallery)

    X_probe_raw, y_probe = pipeline.load_dataset(
        probe_dir, max_per_identity=args.max_probe, verbose=False
    )
    if len(X_probe_raw) == 0:
        return {"rank_1": 0, "eer": 1.0, "auc": 0.5, "d_prime": 0.0}

    X_probe_pca = pipeline.reducer.transform(X_probe_raw)
    top_labels, _ = pipeline.classifier.predict_top_k(X_probe_pca, k=10)

    results = full_evaluation_report(
        y_true=y_probe,
        y_pred_top_k=top_labels,
        X_probe=X_probe_pca,
        y_probe=y_probe,
    )
    return results


def main():
    args = parse_args()
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*65}")
    print(f"  UAV Face Recognition — Ablation Study")
    print(f"  Face Recognition | {timestamp}")
    print(f"{'='*65}\n")

    all_results = {}
    cmc_data = {}

    for config_name, modalities in ABLATION_CONFIGS.items():
        print(f"\n[Ablation] Running: {config_name} | Modalities: {modalities}")
        try:
            res = run_single_ablation(
                modalities, args.gallery, args.probe, args
            )
            all_results[config_name] = res
            cmc_data[config_name] = res.get("cmc_curve", np.zeros(10))
            print(f"  -> Rank-1: {res['rank_1']*100:.2f}% | "
                  f"EER: {res['eer']*100:.2f}% | "
                  f"AUC: {res['auc']:.4f} | "
                  f"d': {res['d_prime']:.4f}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            all_results[config_name] = {"rank_1": 0.0, "eer": 1.0, "auc": 0.5, "d_prime": 0.0}

    # -------------------------------------------------------------------------
    # Print Summary Table
    # -------------------------------------------------------------------------
    print(f"\n{'='*75}")
    print(f"  ABLATION STUDY SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Configuration':<35} {'Rank-1':>8} {'EER':>8} {'AUC':>8} {'d-prime':>9}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
    for name, res in all_results.items():
        print(f"  {name:<35} {res['rank_1']*100:>7.2f}% {res['eer']*100:>7.2f}% "
              f"{res['auc']:>8.4f} {res['d_prime']:>9.4f}")
    print(f"{'='*75}\n")

    # -------------------------------------------------------------------------
    # Save and Plot
    # -------------------------------------------------------------------------
    save_data = {}
    for name, res in all_results.items():
        save_data[name] = {
            k: float(v) if isinstance(v, (float, np.floating)) else
               v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in res.items()
            if k not in ["roc_fpr", "roc_tpr"]
        }

    with open(results_dir / f"ablation_{timestamp}.json", "w") as f:
        json.dump(save_data, f, indent=2)

    plot_ablation_study(
        all_results,
        metric="rank_1",
        title="Ablation Study — Rank-1 Accuracy per Feature Combination",
        save_path=results_dir / f"ablation_rank1_{timestamp}.png",
    )

    plot_ablation_study(
        all_results,
        metric="eer",
        title="Ablation Study — Equal Error Rate per Feature Combination",
        save_path=results_dir / f"ablation_eer_{timestamp}.png",
    )

    valid_cmc = {k: v for k, v in cmc_data.items()
                 if isinstance(v, np.ndarray) and len(v) > 0}
    if valid_cmc:
        plot_cmc_curves(
            valid_cmc,
            title="CMC Curves — Ablation Study",
            save_path=results_dir / f"ablation_cmc_{timestamp}.png",
        )

    print(f"[Done] Ablation results saved to: {results_dir}/")


if __name__ == "__main__":
    main()

