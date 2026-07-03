"""
visualizer.py
-------------
WHAT : Publication-quality visualization utilities for face recognition metrics.
WHY  : Research papers and project reports need clear, consistent plots.
       This module provides all charts used across experiment scripts:
         - ROC curve (TAR vs FAR — standard biometric plot)
         - CMC curve (Rank-k identification rate — standard NIST/FRVT format)
         - Score distribution (genuine vs impostor — shows separation quality)
         - Confusion matrix (which identities are confused with which)
         - Pose / altitude accuracy curves (shows degradation with UAV angle)
         - Ablation bar chart (compares all feature combinations)
         - PCA variance curve (diagnostic: how many components are needed)

       WHY DARK THEME: GitHub dark theme colors are used so plots embed
       naturally in documentation, README files, and dark-mode presentations.
       The color palette (#0d1117 background, #58a6ff blue accents) matches
       GitHub's design language for consistent visual identity.

       WHY MATPLOTLIB not Plotly: Matplotlib produces static PNG/PDF files
       that are reproducible and don't require a browser or internet. Research
       figures must be static for submission.

Face Recognition on Skewed UAV Images
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix


# ── Global Dark Theme Style ───────────────────────────────────────────────────
# Applied to ALL figures produced by this module so every plot looks consistent.
# Colors chosen to match GitHub's dark mode UI for documentation embedding.
plt.rcParams.update({
    "figure.facecolor": "#0d1117",  # GitHub dark background (page background)
    "axes.facecolor":   "#161b22",  # Slightly lighter for plot area contrast
    "axes.edgecolor":   "#30363d",  # Subtle border
    "axes.labelcolor":  "#c9d1d9",  # Light gray text
    "xtick.color":      "#c9d1d9",
    "ytick.color":      "#c9d1d9",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",  # Very subtle grid lines
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        11,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

# Color palette for multi-method comparisons (blue, green, red, yellow, purple, light blue)
COLORS = ["#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff", "#79c0ff"]


def save_or_show(fig: plt.Figure, save_path: str | Path | None, dpi: int = 150) -> None:
    """
    Save figure to file or display it interactively.

    WHY dpi=150: High enough for crisp PNG output in papers and README files
    without making files too large (~50–100KB per figure).
    bbox_inches='tight' removes unnecessary whitespace around the plot.
    facecolor preserves the dark background when saving.
    """
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Visualizer] Saved → {path}")
        plt.close(fig)  # Free memory — important in experiment loops that make many plots
    else:
        plt.show()


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curves",
    save_path: str | Path = None,
) -> None:
    """
    Plot ROC (Receiver Operating Characteristic) curves for one or more methods.

    WHY ROC: Standard plot for biometric verification systems.
    X-axis = False Acceptance Rate (FAR): how often impostors are accepted.
    Y-axis = True Acceptance Rate (TAR): how often genuine users are accepted.
    AUC (Area Under Curve) → 1.0 = perfect, 0.5 = random chance.

    Parameters
    ----------
    roc_data : dict mapping method_name → (fpr_array, tpr_array, auc_score)
               Multiple methods can be overlaid on the same plot for comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (method, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        color = COLORS[idx % len(COLORS)]
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"{method} (AUC={auc_val:.4f})")

    # Reference line: a random classifier sits on the diagonal
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.0, label="Random (AUC=0.5)")

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

    WHY CMC: Standard plot for identification systems (closed-set recognition).
    X-axis = Rank k: we look at top-k predictions.
    Y-axis = % of probes where the correct identity is within top-k.
    Rank-1 accuracy = the most important metric (is the top prediction correct?).

    Parameters
    ----------
    cmc_data : dict mapping method_name → cmc_array of shape (max_rank,)
               Multiple methods can be overlaid for ablation comparison.
    """
    fig, ax   = plt.subplots(figsize=(8, 6))
    max_rank  = max(len(v) for v in cmc_data.values())

    for idx, (method, cmc) in enumerate(cmc_data.items()):
        color = COLORS[idx % len(COLORS)]
        ranks = np.arange(1, len(cmc) + 1)
        ax.plot(ranks, cmc * 100, color=color, lw=2.0,
                marker="o", markersize=4, label=method)

    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification Rate (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.5)
    ax.set_xlim([1, max_rank])
    ax.set_ylim([0, 105])
    # Avoid overcrowded x-tick labels by stepping through ranks
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
    """
    Plot overlapping histograms of genuine and impostor cosine similarity scores.

    WHY THIS PLOT: Shows how well-separated the genuine and impostor score
    distributions are. A good system has a large gap between the two peaks.
    Overlap = region where EER occurs and false decisions are made.
    The d' (d-prime) metric quantifies this separation numerically.

    Parameters
    ----------
    genuine_scores  : scores for same-identity pairs (should be high)
    impostor_scores : scores for different-identity pairs (should be low)
    eer             : if provided, marks the Equal Error Rate threshold line
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(genuine_scores,  bins=50, alpha=0.75, color=COLORS[1],
            label=f"Genuine  (n={len(genuine_scores)})")
    ax.hist(impostor_scores, bins=50, alpha=0.75, color=COLORS[2],
            label=f"Impostor (n={len(impostor_scores)})")

    # Mark the EER threshold where FAR = FRR
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
    """
    Plot a confusion matrix heatmap (Rank-1 predictions vs ground truth).

    WHY LIMIT TO 30 CLASSES: With >30 identities, cell labels become
    illegible. For large datasets, the raw confusion matrix JSON is more
    useful than a visual heatmap.

    Bright diagonal = correct classifications.
    Off-diagonal = which identities are confused with which.

    Parameters
    ----------
    max_classes : int — truncate to first N classes if more exist
    """
    import sklearn.preprocessing as sp  # Lazy import — only needed here

    classes = sorted(set(list(y_true) + list(y_pred)))

    # Truncate if too many classes for a readable plot
    if len(classes) > max_classes:
        classes = classes[:max_classes]
        mask_true = np.isin(y_true, classes)
        mask_pred = np.isin(y_pred, classes)
        mask   = mask_true & mask_pred
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(max(8, len(classes) // 2), max(6, len(classes) // 2)))
    sns.heatmap(
        cm,
        annot=len(classes) <= 20,  # Show cell counts only if readable (≤20 classes)
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
        cbar=True,
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
    Bar chart of Rank-1 accuracy across pose angle or altitude bins.

    Used by run_pose_study.py to show how accuracy degrades as the
    UAV angle increases. The 70% threshold line marks the minimum
    acceptable performance level.

    Parameters
    ----------
    pose_results : dict mapping bin_label → rank_1 accuracy (0–1)
    """
    labels = list(pose_results.keys())
    values = [v * 100 for v in pose_results.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=COLORS[0], alpha=0.85, width=0.6)

    # Add percentage label on top of each bar for easy reading
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9, color="#e6edf3"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rank-1 Identification Rate (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.set_ylim([0, 110])
    # Reference threshold line — arbitrary but useful visual anchor
    ax.axhline(y=70, color=COLORS[2], lw=1.0, linestyle="--",
               alpha=0.7, label="70% threshold")
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
    Horizontal bar chart comparing all feature modality combinations.

    WHY HORIZONTAL BARS: Long combination names ('HOG + LBP + ArcFace') don't
    fit on a vertical x-axis without rotation. Horizontal layout is more readable.
    invert_yaxis() places the best-performing combination at the top.

    Parameters
    ----------
    ablation_results : dict mapping experiment_name → results_dict
                       (from run_ablation.py)
    metric : which metric key to plot (default: 'rank_1')
    """
    names  = list(ablation_results.keys())
    values = [ablation_results[n].get(metric, 0) * 100 for n in names]

    fig, ax  = plt.subplots(figsize=(9, max(4, len(names) * 0.7)))
    colors   = [COLORS[i % len(COLORS)] for i in range(len(names))]
    bars     = ax.barh(names, values, color=colors, alpha=0.85, height=0.5)

    # Add percentage label to the right of each bar
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, color="#e6edf3")

    ax.set_xlabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.set_xlim([0, 110])
    ax.grid(True, axis="x", alpha=0.5)
    ax.invert_yaxis()  # Highest performing combination appears at the top
    fig.tight_layout()
    save_or_show(fig, save_path)


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    target_variance: float = 0.95,
    save_path: str | Path = None,
) -> None:
    """
    Plot the cumulative explained variance curve as a function of PCA components.

    WHY THIS PLOT: Diagnostic tool to justify the chosen PCA variance threshold.
    The 'elbow' in the curve shows the point of diminishing returns — adding
    more components gains little additional variance but increases SVM training time.

    Parameters
    ----------
    explained_variance_ratio : per-component variance fractions (from sklearn PCA)
    target_variance : the threshold line to mark (default 0.95 = 95%)
    """
    cumulative   = np.cumsum(explained_variance_ratio) * 100  # Percentage
    n_comp       = len(cumulative)
    threshold_idx = np.searchsorted(cumulative, target_variance * 100)  # First component crossing target

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n_comp + 1), cumulative, color=COLORS[0], lw=2.0)

    # Horizontal threshold line
    ax.axhline(y=target_variance * 100, color=COLORS[2], lw=1.2,
               linestyle="--", label=f"{target_variance*100:.0f}% variance")
    # Vertical line at the number of components needed
    ax.axvline(x=threshold_idx + 1, color=COLORS[3], lw=1.2,
               linestyle="--", label=f"n={threshold_idx+1} components")

    # Shaded area under the curve for visual clarity
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
    Line chart showing how Rank-1 accuracy changes across degradation severity levels.

    Used by run_degradation.py to visualise the accuracy vs. degradation trade-off.
    A steep drop means the pipeline is sensitive to that degradation type.
    A gradual drop means it is robust.

    Parameters
    ----------
    degradation_results : dict mapping severity_label → results_dict
                          Labels should be ordered from mild to severe.
    """
    labels = list(degradation_results.keys())
    rank1  = [degradation_results[k].get("rank_1", 0) * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(labels, rank1, color=COLORS[0], lw=2.5,
            marker="o", markersize=7, markerfacecolor=COLORS[1])
    # Shaded area under the curve emphasises the trend
    ax.fill_between(range(len(labels)), rank1, alpha=0.12, color=COLORS[0])

    ax.set_xlabel(x_label)
    ax.set_ylabel("Rank-1 Identification Rate (%)")
    ax.set_title(title, fontsize=14, fontweight="bold", color="#e6edf3")
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.5)
    fig.tight_layout()
    save_or_show(fig, save_path)
