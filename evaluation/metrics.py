"""
metrics.py
----------
Comprehensive biometric evaluation metrics.

Implements:
  - Rank-1 and Rank-k Identification Rate (CMC)
  - Equal Error Rate (EER)
  - TAR @ FAR = 0.1% / 1% (NIST standard)
  - ROC Curve + AUC
  - d-prime (d') discriminability index
  - Confusion matrix

Follows ISO/IEC 19795 biometric performance testing standards.

Face Recognition on Skewed UAV Images
"""

import numpy as np
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report
)
from scipy.special import erfinv
from scipy.stats import norm


def rank_k_accuracy(
    y_true: list | np.ndarray,
    y_pred_top_k: np.ndarray,
    k: int = 1,
) -> float:
    """
    Compute Rank-k identification accuracy.

    Parameters
    ----------
    y_true      : ground truth identity labels, shape (n_probes,)
    y_pred_top_k: top-k predicted labels, shape (n_probes, k_max)
    k           : rank threshold

    Returns
    -------
    float in [0, 1]
    """
    y_true = np.array(y_true)
    correct = 0
    for i, gt in enumerate(y_true):
        if gt in y_pred_top_k[i, :k]:
            correct += 1
    return correct / len(y_true)


def compute_cmc_curve(
    y_true: list | np.ndarray,
    y_pred_top_k: np.ndarray,
    max_rank: int = 20,
) -> np.ndarray:
    """
    Compute the Cumulative Match Characteristic (CMC) curve.

    Returns
    -------
    np.ndarray of shape (max_rank,) — Rank-k accuracy for k=1..max_rank
    """
    cmc = np.zeros(max_rank)
    for k in range(1, max_rank + 1):
        cmc[k - 1] = rank_k_accuracy(y_true, y_pred_top_k, k)
    return cmc


def compute_eer(y_true_binary: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) for a binary verification task.

    Parameters
    ----------
    y_true_binary : np.ndarray — binary (1=genuine, 0=impostor)
    scores        : np.ndarray — similarity/confidence scores

    Returns
    -------
    (eer, threshold) — EER value and corresponding threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the threshold where FPR ≈ FNR (EER point)
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = thresholds[eer_idx]

    return float(eer), float(threshold)


def compute_tar_at_far(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
    far_target: float = 0.001,
) -> tuple[float, float]:
    """
    Compute TAR (True Accept Rate) at a specified FAR (False Accept Rate).

    Standard: NIST FRVT uses FAR=0.1% (0.001) as primary operating point.

    Parameters
    ----------
    far_target : float — target FAR (e.g., 0.001 for 0.1%)

    Returns
    -------
    (tar, actual_far) — TAR value and actual FAR achieved
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, scores, pos_label=1)

    # Find operating point closest to target FAR
    idx = np.searchsorted(fpr, far_target)
    if idx >= len(fpr):
        idx = len(fpr) - 1

    return float(tpr[idx]), float(fpr[idx])


def compute_roc_auc(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.

    Returns
    -------
    (fpr, tpr, thresholds, auc_score)
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, scores, pos_label=1)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, float(auc_score)


def compute_dprime(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> float:
    """
    Compute d' (d-prime) — signal detection theory discriminability index.

    d' = (μ_genuine - μ_impostor) / sqrt(0.5 * (σ²_genuine + σ²_impostor))

    Higher d' = better separation between genuine and impostor distributions.
    d' > 2.0 is generally considered good for biometric systems.

    Returns
    -------
    float — d' value
    """
    mu_g = np.mean(genuine_scores)
    mu_i = np.mean(impostor_scores)
    var_g = np.var(genuine_scores)
    var_i = np.var(impostor_scores)

    denom = np.sqrt(0.5 * (var_g + var_i))
    if denom < 1e-8:
        return 0.0
    return float((mu_g - mu_i) / denom)


def generate_verification_pairs(
    X: np.ndarray,
    y: list | np.ndarray,
    n_genuine: int = 1000,
    n_impostor: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate genuine and impostor pairs for verification evaluation.

    Returns
    -------
    scores   : cosine similarity scores for all pairs
    labels   : 1 = genuine pair, 0 = impostor pair
    pair_idx : (n_pairs, 2) indices of paired samples
    """
    rng = np.random.default_rng(random_state)
    y = np.array(y)
    n = len(y)

    genuine_pairs, impostor_pairs = [], []

    # Build identity index
    id_to_idx = {}
    for i, label in enumerate(y):
        id_to_idx.setdefault(label, []).append(i)

    # Genuine pairs (same identity)
    identities_with_multiple = [
        k for k, v in id_to_idx.items() if len(v) >= 2
    ]
    for _ in range(n_genuine):
        if not identities_with_multiple:
            break
        identity = rng.choice(identities_with_multiple)
        idxs = id_to_idx[identity]
        i, j = rng.choice(idxs, size=2, replace=False)
        genuine_pairs.append((i, j))

    # Impostor pairs (different identities)
    all_identities = list(id_to_idx.keys())
    for _ in range(n_impostor):
        id_a, id_b = rng.choice(all_identities, size=2, replace=False)
        i = rng.choice(id_to_idx[id_a])
        j = rng.choice(id_to_idx[id_b])
        impostor_pairs.append((i, j))

    all_pairs = genuine_pairs + impostor_pairs
    labels = np.array([1] * len(genuine_pairs) + [0] * len(impostor_pairs))
    pair_idx = np.array(all_pairs)

    # Compute cosine similarity scores
    scores = np.array([
        float(cosine_similarity(X[i], X[j]))
        for i, j in all_pairs
    ])

    return scores, labels, pair_idx


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def full_evaluation_report(
    y_true: list | np.ndarray,
    y_pred_top_k: np.ndarray,
    X_probe: np.ndarray,
    y_probe: list | np.ndarray,
    n_genuine: int = 500,
    n_impostor: int = 500,
) -> dict:
    """
    Run complete evaluation and return results dictionary.

    Parameters
    ----------
    y_true       : ground truth labels for probes
    y_pred_top_k : top-k predictions, shape (n_probes, k_max)
    X_probe      : PCA-transformed probe features (for verification pairs)
    y_probe      : probe labels (for verification pair generation)

    Returns
    -------
    dict with all metrics
    """
    results = {}

    # --- CMC ---
    cmc = compute_cmc_curve(y_true, y_pred_top_k, max_rank=20)
    results["rank_1"]  = float(cmc[0])
    results["rank_5"]  = float(cmc[4]) if len(cmc) >= 5 else float(cmc[-1])
    results["rank_10"] = float(cmc[9]) if len(cmc) >= 10 else float(cmc[-1])
    results["cmc_curve"] = cmc

    # --- Verification Metrics (on probe set) ---
    scores, bin_labels, _ = generate_verification_pairs(
        X_probe, y_probe, n_genuine=n_genuine, n_impostor=n_impostor
    )

    eer, eer_thresh = compute_eer(bin_labels, scores)
    results["eer"] = eer

    tar_01, far_01 = compute_tar_at_far(bin_labels, scores, far_target=0.001)
    results["tar_at_far_0.1%"] = tar_01

    tar_1, far_1 = compute_tar_at_far(bin_labels, scores, far_target=0.01)
    results["tar_at_far_1%"] = tar_1

    fpr, tpr, _, auc_score = compute_roc_auc(bin_labels, scores)
    results["auc"] = auc_score
    results["roc_fpr"] = fpr
    results["roc_tpr"] = tpr

    # --- d-prime ---
    genuine_scores  = scores[bin_labels == 1]
    impostor_scores = scores[bin_labels == 0]
    results["d_prime"] = compute_dprime(genuine_scores, impostor_scores)

    return results


def print_report(results: dict, label: str = "Full Pipeline") -> None:
    """Print a formatted metrics summary."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT: {label}")
    print(f"{'='*60}")
    print(f"  Rank-1  Accuracy     : {results.get('rank_1', 0)*100:.2f}%")
    print(f"  Rank-5  Accuracy     : {results.get('rank_5', 0)*100:.2f}%")
    print(f"  Rank-10 Accuracy     : {results.get('rank_10', 0)*100:.2f}%")
    print(f"  EER                  : {results.get('eer', 0)*100:.2f}%")
    print(f"  TAR @ FAR=0.1%       : {results.get('tar_at_far_0.1%', 0)*100:.2f}%")
    print(f"  TAR @ FAR=1%         : {results.get('tar_at_far_1%', 0)*100:.2f}%")
    print(f"  AUC (ROC)            : {results.get('auc', 0):.4f}")
    print(f"  d-prime (d')         : {results.get('d_prime', 0):.4f}")
    print(f"{'='*60}\n")

