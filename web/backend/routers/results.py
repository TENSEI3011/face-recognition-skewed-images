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


def _load_latest_metrics(exp_dir: Path) -> Optional[dict]:
    """Load the most recent metrics_*.json in an experiment directory."""
    jsons = sorted(exp_dir.glob("metrics_*.json"))
    if not jsons:
        return None
    with open(jsons[-1]) as f:
        return json.load(f)


def _load_plots(exp_dir: Path) -> list:
    """Return base64-encoded PNG plots from an experiment directory."""
    plots = []
    pngs  = sorted(exp_dir.glob("*.png"))
    for png in pngs:
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
