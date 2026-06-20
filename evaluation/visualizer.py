"""
visualizer.py
-------------
Visualization utilities for face recognition evaluation results.

Generates:
  - ROC curves (single and multi-method comparison)
  - CMC curves
  - Confusion matrix heatmap
  - Rank-1 accuracy vs pose angle bar chart
  - Rank-1 accuracy vs altitude line chart
  - Score distribution (genuine vs impostor)
  - PCA variance explained curve
  - Ablation study bar chart

Face Recognition on Skewed UAV Images
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


# Global style settings
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#c9d1d9",
    "ytick.color":      "#c9d1d9",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

COLORS = ["#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff", "#79c0ff"]


def save_or_show(fig: plt.Figure, save_path: str | Path | None, dpi: int = 150) -> None:
    """Save figure to file or display it."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Visualizer] Saved → {path}")
        plt.close(fig)
    else:
        plt.show()


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curves",
    save_path: str | Path = None,
) -> None:
    """
    Plot ROC curves for multiple methods.

    Parameters
    ----------
    roc_data : dict mapping method_name → (fpr, tpr, auc_score)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (method, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        color = COLORS[idx % len(COLORS)]
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"{method} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.0, label="Random")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (TAR)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_cmc_curves(
    cmc_data: dict[str, np.ndarray],
    title: str = "CMC Curves",
    save_path: str | Path = None,
) -> None:
    """
    Plot CMC (Cumulative Match Characteristic) curves.

    Parameters
    ----------
    cmc_data : dict mapping method_name → cmc_array (length = max_rank)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    max_rank = max(len(v) for v in cmc_data.values())

    for idx, (method, cmc) in enumerate(cmc_data.items()):
        color = COLORS[idx % len(COLORS)]
        ranks = np.arange(1, len(cmc) + 1)
        ax.plot(ranks, cmc * 100, color=color, lw=2.0, marker="o",
                markersize=4, label=method)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification Rate (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.5)
    ax.set_xlim([1, max_rank])
    ax.set_ylim([0, 105])
    ax.set_xticks(range(1, max_rank + 1, max(1, max_rank // 10)))
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_score_distribution(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    eer: float = None,
    title: str = "Genuine vs Impostor Score Distribution",
    save_path: str | Path = None,
) -> None:
    """Plot distribution of genuine and impostor similarity scores."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(genuine_scores,  bins=50, alpha=0.75, color=COLORS[1],
            label=f"Genuine (n={len(genuine_scores)})")
    ax.hist(impostor_scores, bins=50, alpha=0.75, color=COLORS[2],
            label=f"Impostor (n={len(impostor_scores)})")

    if eer is not None:
        ax.axvline(x=eer, color="#d29922", lw=1.5, linestyle="--",
                   label=f"EER threshold ≈ {eer:.3f}")

    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.legend()
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_confusion_matrix(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | Path = None,
    max_classes: int = 30,
) -> None:
    """Plot confusion matrix heatmap (truncated for large class counts)."""
    import sklearn.preprocessing as sp

    classes = sorted(set(list(y_true) + list(y_pred)))
    if len(classes) > max_classes:
        classes = classes[:max_classes]
        mask_true = np.isin(y_true, classes)
        mask_pred = np.isin(y_pred, classes)
        mask = mask_true & mask_pred
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(max(8, len(classes) // 2), max(6, len(classes) // 2)))
    sns.heatmap(
        cm, annot=len(classes) <= 20, fmt="d",
        cmap="Blues", xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5, cbar=True,
        annot_kws={"size": 8},
    )
    ax.set_xlabel("Predicted Identity")
    ax.set_ylabel("True Identity")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_rank1_vs_pose(
    pose_results: dict[str, float],
    title: str = "Rank-1 Accuracy vs Pose Angle",
    xlabel: str = "Pose Angle (degrees)",
    save_path: str | Path = None,
) -> None:
    """
    Bar/line chart of Rank-1 accuracy across pose angles or altitude bands.

    Parameters
    ----------
    pose_results : dict mapping angle/altitude label → rank-1 float
    """
    labels = list(pose_results.keys())
    values = [v * 100 for v in pose_results.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=COLORS[0], alpha=0.85, width=0.6)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9, color="#e6edf3")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rank-1 Identification Rate (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.set_ylim([0, 110])
    ax.axhline(y=70, color=COLORS[2], lw=1.0, linestyle="--", alpha=0.7, label="70% threshold")
    ax.grid(True, axis="y", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_ablation_study(
    ablation_results: dict[str, dict],
    metric: str = "rank_1",
    title: str = "Ablation Study — Feature Modality Contribution",
    save_path: str | Path = None,
) -> None:
    """
    Horizontal bar chart showing contribution of each feature modality.

    Parameters
    ----------
    ablation_results : dict mapping experiment_name → results_dict
    metric : key in results_dict to plot
    """
    names  = list(ablation_results.keys())
    values = [ablation_results[n].get(metric, 0) * 100 for n in names]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.7)))
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    bars = ax.barh(names, values, color=colors, alpha=0.85, height=0.5)

    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color="#e6edf3")

    ax.set_xlabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.set_xlim([0, 110])
    ax.grid(True, axis="x", alpha=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    target_variance: float = 0.95,
    save_path: str | Path = None,
) -> None:
    """Plot cumulative explained variance vs number of PCA components."""
    cumulative = np.cumsum(explained_variance_ratio) * 100
    n_comp = len(cumulative)
    threshold_idx = np.searchsorted(cumulative, target_variance * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n_comp + 1), cumulative, color=COLORS[0], lw=2.0)
    ax.axhline(y=target_variance * 100, color=COLORS[2], lw=1.2,
               linestyle="--", label=f"{target_variance*100:.0f}% variance")
    ax.axvline(x=threshold_idx + 1, color=COLORS[3], lw=1.2,
               linestyle="--", label=f"n={threshold_idx+1} components")
    ax.fill_between(range(1, n_comp + 1), cumulative, alpha=0.15, color=COLORS[0])
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA — Cumulative Explained Variance", fontsize=14,
                 fontweight="bold", color="#e6edf3")
    ax.legend()
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_degradation_curve(
    degradation_results: dict[str, dict],
    x_label: str = "Degradation Level",
    title: str = "Rank-1 Accuracy Under Degradation",
    save_path: str | Path = None,
) -> None:
    """
    Line chart: Rank-1 accuracy vs degradation severity.

    Parameters
    ----------
    degradation_results : dict mapping severity_label → results_dict
    """
    labels = list(degradation_results.keys())
    rank1  = [degradation_results[k].get("rank_1", 0) * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(labels, rank1, color=COLORS[0], lw=2.5, marker="o",
            markersize=7, markerfacecolor=COLORS[1])
    ax.fill_between(range(len(labels)), rank1, alpha=0.12, color=COLORS[0])

    ax.set_xlabel(x_label)
    ax.set_ylabel("Rank-1 Identification Rate (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    save_or_show(fig, save_path)

