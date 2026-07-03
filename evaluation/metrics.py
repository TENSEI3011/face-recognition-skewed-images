"""
metrics.py
----------
WHAT : Comprehensive biometric evaluation metrics module.
WHY  : Standard accuracy (correct/total) is insufficient for biometric systems.
       Biometric evaluation uses specialized metrics from ISO/IEC 19795 and NIST
       FRVT standards that distinguish between:
         - IDENTIFICATION: who is this person? (closed-set, CMC curves)
         - VERIFICATION: is this person who they claim to be? (EER, TAR@FAR, ROC)

       METRICS IMPLEMENTED:
         Rank-k IR   — Identification Rate at rank k (CMC curve point)
                       "Is the correct identity within the top-k predictions?"
         EER         — Equal Error Rate: the FAR at which FRR = FAR.
                       Lower = better. 0% = perfect; 50% = random.
         TAR@FAR=0.1% — True Accept Rate when False Accept Rate = 0.1%
                        (NIST FRVT primary operating point for high-security applications)
         AUC         — Area under ROC curve. 1.0 = perfect; 0.5 = random.
         d' (d-prime) — Signal detection theory discriminability index.
                        Measures separation between genuine/impostor distributions.
                        d' > 2.0 = good system; d' < 1.0 = poor system.

       WHY COSINE SIMILARITY FOR VERIFICATION SCORES:
         PCA features are whitened and unit-normalized. In this space, cosine
         similarity (dot product) directly measures angular distance in the
         identity embedding space, which is what SVM decision boundaries optimize.

Follows ISO/IEC 19795 biometric performance testing standards.

Face Recognition on Skewed UAV Images
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from scipy.special import erfinv
from scipy.stats import norm


def rank_k_accuracy(
    y_true: list | np.ndarray,
    y_pred_top_k: np.ndarray,
    k: int = 1,
) -> float:
    """
    Compute Rank-k Identification Rate (IR).

    WHY RANK-k vs TOP-1 ACCURACY:
      Top-1 accuracy penalises the system even when the correct identity is the
      2nd most likely prediction. Rank-5 accuracy tells us: "even if the system
      isn't sure, does it at least strongly suspect the right person?"
      This matters in human-in-the-loop surveillance where an operator reviews
      the top-k candidates.

    Parameters
    ----------
    y_true       : ground truth labels, shape (n_probes,)
    y_pred_top_k : predicted labels ranked by confidence, shape (n_probes, k_max)
    k            : rank threshold (1 = only top prediction counts)

    Returns
    -------
    float in [0, 1] — fraction of probes where correct identity is in top-k
    """
    y_true  = np.array(y_true)
    correct = 0
    for i, gt in enumerate(y_true):
        if gt in y_pred_top_k[i, :k]:  # Is ground truth in the top-k predictions?
            correct += 1
    return correct / len(y_true)


def compute_cmc_curve(
    y_true: list | np.ndarray,
    y_pred_top_k: np.ndarray,
    max_rank: int = 20,
) -> np.ndarray:
    """
    Compute the full Cumulative Match Characteristic (CMC) curve.

    The CMC curve shows Rank-k IR for every rank from 1 to max_rank.
    It is monotonically non-decreasing (higher rank = at least as good as lower).

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
    Compute Equal Error Rate (EER) for binary verification.

    WHY EER: EER is the operating point where FAR = FRR (False Reject Rate).
    It gives a single-number summary of system performance independent of
    threshold choice. Systems with EER < 5% are considered good for real use.

    HOW: EER is the point on the ROC curve where FPR = FNR = 1 - TPR.
    We find this point by minimizing |FPR - FNR| across all thresholds.

    Parameters
    ----------
    y_true_binary : np.ndarray — 1 for genuine pairs, 0 for impostor pairs
    scores        : np.ndarray — similarity scores (higher = more similar)

    Returns
    -------
    (eer, threshold) — EER value in [0,1] and the score threshold at EER point
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, scores, pos_label=1)
    fnr = 1 - tpr  # False Negative Rate = 1 - True Positive Rate

    # EER occurs where FPR ≈ FNR — find the closest point
    eer_idx   = np.nanargmin(np.abs(fpr - fnr))
    # Average FPR and FNR at that point for a symmetric estimate
    eer       = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = thresholds[eer_idx]

    return float(eer), float(threshold)


def compute_tar_at_far(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
    far_target: float = 0.001,
) -> tuple[float, float]:
    """
    Compute TAR (True Accept Rate) at a target FAR (False Accept Rate).

    WHY TAR@FAR=0.1%: This is the NIST FRVT primary evaluation point for
    high-security systems (access control, passport verification). At 0.1% FAR,
    only 1 in 1000 impostors gets through — this is the operational constraint.
    Higher TAR at this constraint = more genuine users served correctly.

    Parameters
    ----------
    far_target : float — e.g., 0.001 for 0.1%, 0.01 for 1%

    Returns
    -------
    (tar, actual_far) — TAR achieved and actual FAR at that operating point
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, scores, pos_label=1)

    # Find the ROC point where FAR is closest to (but does not exceed) far_target
    idx = np.searchsorted(fpr, far_target)
    if idx >= len(fpr):
        idx = len(fpr) - 1

    return float(tpr[idx]), float(fpr[idx])


def compute_roc_auc(
    y_true_binary: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC (Area Under Curve).

    WHY AUC: AUC summarises the full ROC curve in one number.
    AUC = probability that a genuine pair scores higher than a random impostor pair.
    Threshold-independent metric — useful for comparing systems without
    committing to a specific operating point.

    Returns
    -------
    (fpr, tpr, thresholds, auc_score)
    """
    fpr, tpr, thresholds = roc_curve(y_true_binary, scores, pos_label=1)
    auc_score = auc(fpr, tpr)  # Area under curve using trapezoidal integration
    return fpr, tpr, thresholds, float(auc_score)


def compute_dprime(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> float:
    """
    Compute d' (d-prime) — signal detection theory discriminability index.

    FORMULA:
      d' = (μ_genuine - μ_impostor) / sqrt(0.5 × (σ²_genuine + σ²_impostor))

    WHY d': Assumes genuine and impostor scores follow Gaussian distributions.
    Measures how many standard deviations apart the two distributions are.
      d' = 0   → complete overlap (random chance)
      d' = 1   → some separation
      d' = 2   → good system
      d' = 3+  → excellent system

    WHY pooled standard deviation: Using a pooled variance (average of both)
    rather than just one distribution makes d' symmetric and more robust when
    the two distributions have different spreads.

    Returns
    -------
    float — d' value (higher = better separation)
    """
    mu_g  = np.mean(genuine_scores)
    mu_i  = np.mean(impostor_scores)
    var_g = np.var(genuine_scores)
    var_i = np.var(impostor_scores)

    # Pooled standard deviation — the common signal detection formula
    denom = np.sqrt(0.5 * (var_g + var_i))

    if denom < 1e-8:
        return 0.0  # Both distributions are point masses — degenerate case

    return float((mu_g - mu_i) / denom)


def generate_verification_pairs(
    X: np.ndarray,
    y: list | np.ndarray,
    n_genuine: int = 1000,
    n_impostor: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample genuine and impostor pairs from a probe set for verification evaluation.

    WHY SAMPLING instead of all pairs: Exhaustive pairing is O(n²) — with 1000
    probe samples, that's 500k pairs. Sampling 500 genuine + 500 impostor pairs
    is statistically sufficient and much faster.

    WHY FIXED random_state=42: Ensures reproducibility across runs so metric
    values are comparable between experiments.

    Parameters
    ----------
    X           : PCA-projected feature matrix, shape (n_samples, n_components)
    y           : identity labels for each sample
    n_genuine   : number of same-identity pairs to sample
    n_impostor  : number of different-identity pairs to sample
    random_state : RNG seed for reproducibility

    Returns
    -------
    scores   : cosine similarity for each pair, shape (n_genuine + n_impostor,)
    labels   : 1 = genuine, 0 = impostor
    pair_idx : (n_pairs, 2) array of sample index pairs
    """
    rng = np.random.default_rng(random_state)
    y   = np.array(y)

    genuine_pairs, impostor_pairs = [], []

    # Build a lookup: identity → list of sample indices with that identity
    id_to_idx = {}
    for i, label in enumerate(y):
        id_to_idx.setdefault(label, []).append(i)

    # ── Sample GENUINE pairs (same identity) ──────────────────────────────────
    # Only identities with ≥2 samples can contribute a genuine pair
    identities_with_multiple = [k for k, v in id_to_idx.items() if len(v) >= 2]
    for _ in range(n_genuine):
        if not identities_with_multiple:
            break
        identity = rng.choice(identities_with_multiple)
        idxs     = id_to_idx[identity]
        i, j     = rng.choice(idxs, size=2, replace=False)  # Two DIFFERENT samples
        genuine_pairs.append((i, j))

    # ── Sample IMPOSTOR pairs (different identities) ───────────────────────────
    all_identities = list(id_to_idx.keys())
    if len(all_identities) >= 2:
        for _ in range(n_impostor):
            id_a, id_b = rng.choice(all_identities, size=2, replace=False)  # Different identities
            i = rng.choice(id_to_idx[id_a])
            j = rng.choice(id_to_idx[id_b])
            impostor_pairs.append((i, j))

    all_pairs = genuine_pairs + impostor_pairs
    labels    = np.array([1] * len(genuine_pairs) + [0] * len(impostor_pairs))
    pair_idx  = np.array(all_pairs)

    # Compute cosine similarity for each pair — our verification score
    scores = np.array([
        float(cosine_similarity(X[i], X[j]))
        for i, j in all_pairs
    ])

    return scores, labels, pair_idx


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.

    WHY COSINE instead of Euclidean distance: PCA-whitened features lie on
    a hypersphere (unit variance per dimension). Cosine similarity measures
    the angle between vectors — invariant to magnitude scaling.
    Range: [-1, 1] where 1 = identical direction, 0 = orthogonal, -1 = opposite.

    Returns 0.0 for zero vectors (degenerate case from failed feature extraction).
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0  # Zero vector = no information, treat as completely unrelated
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
    Run the complete evaluation suite and return all metrics in a dictionary.

    WHY ONE FUNCTION: Experiment scripts (run_baseline.py, run_ablation.py)
    all need the same metrics. This centralizes the computation so metrics
    are consistently computed everywhere.

    Parameters
    ----------
    y_true       : ground truth probe labels, shape (n_probes,)
    y_pred_top_k : top-k predictions from classifier, shape (n_probes, k_max)
    X_probe      : PCA-projected probe features (for verification pair scoring)
    y_probe      : probe labels (for sampling genuine/impostor pairs)
    n_genuine    : number of genuine pairs to sample for verification metrics
    n_impostor   : number of impostor pairs to sample

    Returns
    -------
    dict with keys: rank_1, rank_5, rank_10, cmc_curve,
                    eer, tar_at_far_0.1%, tar_at_far_1%,
                    auc, roc_fpr, roc_tpr, d_prime
    """
    results = {}

    # ── Identification Metrics (CMC) ──────────────────────────────────────────
    cmc = compute_cmc_curve(y_true, y_pred_top_k, max_rank=20)
    results["rank_1"]    = float(cmc[0])
    results["rank_5"]    = float(cmc[4]) if len(cmc) >= 5 else float(cmc[-1])
    results["rank_10"]   = float(cmc[9]) if len(cmc) >= 10 else float(cmc[-1])
    results["cmc_curve"] = cmc

    # ── Verification Metrics (require ≥2 identities in probe) ─────────────────
    n_identities_in_probe = len(set(y_probe))
    scores, bin_labels, _ = generate_verification_pairs(
        X_probe, y_probe, n_genuine=n_genuine, n_impostor=n_impostor
    )

    if len(scores) == 0 or n_identities_in_probe < 2:
        # Cannot compute verification metrics with only one identity
        # Return NaN so calling code can handle gracefully
        results["eer"]             = float("nan")
        results["tar_at_far_0.1%"] = float("nan")
        results["tar_at_far_1%"]   = float("nan")
        results["roc_fpr"]         = np.array([0.0, 1.0])
        results["roc_tpr"]         = np.array([0.0, 1.0])
        results["auc"]             = float("nan")
        results["d_prime"]         = float("nan")
        return results

    # EER
    eer, _             = compute_eer(bin_labels, scores)
    results["eer"]     = eer

    # TAR at two standard NIST operating points
    tar_01, _                    = compute_tar_at_far(bin_labels, scores, far_target=0.001)
    results["tar_at_far_0.1%"]   = tar_01

    tar_1, _                     = compute_tar_at_far(bin_labels, scores, far_target=0.01)
    results["tar_at_far_1%"]     = tar_1

    # ROC curve and AUC
    fpr, tpr, _, auc_score = compute_roc_auc(bin_labels, scores)
    results["auc"]     = auc_score
    results["roc_fpr"] = fpr
    results["roc_tpr"] = tpr

    # d-prime (discriminability)
    genuine_scores      = scores[bin_labels == 1]
    impostor_scores     = scores[bin_labels == 0]
    results["d_prime"]  = compute_dprime(genuine_scores, impostor_scores)

    return results


def print_report(results: dict, label: str = "Full Pipeline") -> None:
    """
    Print a formatted, human-readable summary of all evaluation metrics.
    Called by experiment scripts after full_evaluation_report().
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT: {label}")
    print(f"{'='*60}")
    print(f"  Rank-1  Accuracy     : {results.get('rank_1',  0) * 100:.2f}%")
    print(f"  Rank-5  Accuracy     : {results.get('rank_5',  0) * 100:.2f}%")
    print(f"  Rank-10 Accuracy     : {results.get('rank_10', 0) * 100:.2f}%")
    print(f"  EER                  : {results.get('eer',     0) * 100:.2f}%")
    print(f"  TAR @ FAR=0.1%       : {results.get('tar_at_far_0.1%', 0) * 100:.2f}%")
    print(f"  TAR @ FAR=1%         : {results.get('tar_at_far_1%',   0) * 100:.2f}%")
    print(f"  AUC (ROC)            : {results.get('auc',     0):.4f}")
    print(f"  d-prime (d')         : {results.get('d_prime', 0):.4f}")
    print(f"{'='*60}\n")
