"""
results.py — Results & plots router
Serves pre-computed experiment plots (PNG) and metrics (JSON) from results/ directory.
"""

import json
import base64
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

from web.backend.config import RESULTS_DIR

router = APIRouter(prefix="/api/results", tags=["results"])

EXPERIMENT_DIRS = {
    "baseline":    RESULTS_DIR / "baseline",
    "ablation":    RESULTS_DIR / "ablation",
    "pose_study":  RESULTS_DIR / "pose_study",
    "degradation": RESULTS_DIR / "degradation",
}

PLOT_PREFIXES = ["cmc", "roc", "confusion", "pca_variance", "score_dist",
                 "ablation_rank1", "ablation_eer", "pose_rank1", "degradation"]


def _normalize_metrics(raw: dict) -> dict:
    """
    Normalize metric keys from experiment JSON format to frontend-expected format.
    Experiment scripts save: rank_1, rank_5, tar_at_far_0.1%
    Frontend JS expects:     rank1,  rank5,  tar_at_far
    """
    normalized = dict(raw)  # copy

    # Map underscore variants → frontend keys (if not already present)
    key_map = {
        "rank_1":          "rank1",
        "rank_5":          "rank5",
        "rank_10":         "rank10",
        "tar_at_far_0.1%": "tar_at_far",
        "tar_at_far_1%":   "tar_at_far_1",
    }
    for src, dst in key_map.items():
        if src in normalized and dst not in normalized:
            normalized[dst] = normalized[src]

    return normalized


def _load_latest_metrics(exp_dir: Path) -> Optional[dict]:
    """Load the most recent metrics_*.json (or ablation_*.json) in an experiment directory."""
    # Standard metrics files
    jsons = sorted(exp_dir.glob("metrics_*.json"))
    if jsons:
        with open(jsons[-1]) as f:
            return _normalize_metrics(json.load(f))
    # Ablation study saves as ablation_*.json with nested structure
    ablation_jsons = sorted(exp_dir.glob("ablation_*.json"))
    if ablation_jsons:
        with open(ablation_jsons[-1]) as f:
            raw = json.load(f)
        # raw is { config_name: { rank_1, eer, ... }, ... }
        # Find the best-performing config (highest rank_1)
        best_conf = max(raw, key=lambda k: raw[k].get("rank_1", 0), default=None)
        if best_conf:
            m = raw[best_conf]
            return {
                "rank1":      m.get("rank_1"),
                "rank5":      m.get("rank_5"),
                "eer":        m.get("eer"),
                "auc":        m.get("auc"),
                "tar_at_far": m.get("tar_at_far_0.1%"),
                "d_prime":    m.get("d_prime"),
                "_note":      f"Best config: {best_conf}",
                "_all":       {
                    k: {"rank1": v.get("rank_1"), "eer": v.get("eer")}
                    for k, v in raw.items()
                },
            }
    return None


def _load_plots(exp_dir: Path) -> list:
    """Return base64-encoded PNG plots from an experiment directory.

    Deduplicates by plot type: each run saves timestamped files like
    'cmc_20260625_003906.png', 'cmc_20260626_233012.png', etc.
    Only the most recent file per plot type (prefix) is returned so the
    UI shows exactly one CMC curve, one ROC curve, one confusion matrix, etc.
    """
    import re

    def _plot_type(stem: str) -> str:
        """
        Extract the plot type prefix from a timestamped filename.
        'cmc_20260625_003906'   → 'cmc'
        'confusion_20260702'    → 'confusion'
        'pca_variance_20260630' → 'pca_variance'
        'ablation_rank1_...'    → 'ablation_rank1'
        """
        # Split on the first segment that looks like an 8-digit date (YYYYMMDD)
        parts = re.split(r'_\d{8}', stem)
        return parts[0].strip('_')

    # Group all PNGs by their plot type, keeping the latest (alphabetically = latest timestamp)
    type_to_png: dict[str, Path] = {}
    for png in sorted(exp_dir.glob("*.png")):
        ptype = _plot_type(png.stem)
        # sorted() gives ascending order so later iteration = newer file → keeps latest
        type_to_png[ptype] = png

    # Build result in a stable display order
    DISPLAY_ORDER = [
        "cmc", "roc", "confusion", "score_dist", "pca_variance",
        "ablation_rank1", "ablation_eer", "pose_rank1", "degradation",
    ]

    ordered_types = []
    for t in DISPLAY_ORDER:
        if t in type_to_png:
            ordered_types.append(t)
    # Append any types not in the explicit order list
    for t in type_to_png:
        if t not in ordered_types:
            ordered_types.append(t)

    plots = []
    for ptype in ordered_types:
        png = type_to_png[ptype]
        try:
            data = png.read_bytes()
            b64  = base64.b64encode(data).decode()
            plots.append({
                "filename": png.name,
                "title":    _pretty_title(png.stem),
                "data":     f"data:image/png;base64,{b64}",
            })
        except Exception:
            pass
    return plots


def _pretty_title(stem: str) -> str:
    """Convert filename stem like 'cmc_20260702_005014' → 'CMC Curve'."""
    prefix = stem.split("_")[0].upper()
    labels = {
        "CMC":          "CMC Curve (Rank-k Identification Rate)",
        "ROC":          "ROC Curve (TAR vs FAR)",
        "CONFUSION":    "Confusion Matrix",
        "PCA":          "PCA Variance Explained",
        "SCORE":        "Score Distribution",
        "ABLATION":     "Ablation Study Results",
        "POSE":         "Pose-Stratified Evaluation",
        "DEGRADATION":  "Degradation Sweep Results",
    }
    for key, label in labels.items():
        if stem.upper().startswith(key):
            return label
    return stem.replace("_", " ").title()


@router.get("")
def list_experiments():
    """List available experiment result directories and whether they have data."""
    result = {}
    for name, exp_dir in EXPERIMENT_DIRS.items():
        has_plots   = exp_dir.exists() and bool(list(exp_dir.glob("*.png")))
        has_metrics = exp_dir.exists() and bool(list(exp_dir.glob("metrics_*.json")))
        result[name] = {
            "has_plots":   has_plots,
            "has_metrics": has_metrics,
            "plot_count":  len(list(exp_dir.glob("*.png"))) if exp_dir.exists() else 0,
        }
    return result


@router.get("/{experiment}")
def get_experiment_results(experiment: str):
    """Get all plots and latest metrics for an experiment."""
    if experiment not in EXPERIMENT_DIRS:
        raise HTTPException(status_code=404, detail=f"Unknown experiment: {experiment}")

    exp_dir = EXPERIMENT_DIRS[experiment]
    if not exp_dir.exists():
        return {"experiment": experiment, "plots": [], "metrics": None,
                "message": "No results yet. Run the experiment to generate results."}

    plots   = _load_plots(exp_dir)
    metrics = _load_latest_metrics(exp_dir)

    return {
        "experiment": experiment,
        "plots":      plots,
        "metrics":    metrics,
        "plot_count": len(plots),
    }
