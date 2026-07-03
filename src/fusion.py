"""
fusion.py
---------
WHAT : Feature fusion module — combines HOG, LBP, Geometry, and ArcFace
       descriptors into a single vector ready for PCA + SVM.
WHY  : Each feature modality captures a different aspect of face appearance:
         HOG      → edge/gradient structure (shape-based)
         LBP      → local texture patterns (illumination-robust)
         Geometry → landmark ratios (illumination-invariant, pose-sensitive)
         ArcFace  → deep identity embedding (generalises across conditions)
       Combining them exploits complementary information — when one modality
       fails (e.g., ArcFace struggles at extreme pose), others compensate.
       This is the key advantage of the hybrid approach over single-modality systems.

       FUSION STRATEGY — L2-normalize then concatenate:
         WHY NOT SUM/AVERAGE: Different modalities have different feature dimensions
           (HOG=3780, LBP=512, Geometry=159, ArcFace=512). Averaging dimensions
           of different sizes is meaningless.
         WHY L2-NORMALIZE FIRST: Without normalization, high-dimensional vectors
           (HOG=3780-D) would dominate the Euclidean distance even if ArcFace
           (512-D) is more discriminative. L2 normalization makes each modality
           contribute with equal magnitude before concatenation.
         WHY CONCATENATE: Preserves all information — no information lost to
           projection or averaging. PCA downstream will learn to weight them optimally.

Pipeline position: feature extractors → [FeatureFusion] → PCAReducer → SVM

Face Recognition on Skewed UAV Images
"""

import numpy as np
from typing import Optional


# ── Modality name constants ───────────────────────────────────────────────────
# Using named constants avoids typo bugs when specifying modalities as strings
MODALITY_HOG      = "hog"
MODALITY_LBP      = "lbp"
MODALITY_GEOMETRY = "geometry"
MODALITY_ARCFACE  = "arcface"

# All supported modalities — used as the default set and for ablation validation
ALL_MODALITIES = [MODALITY_HOG, MODALITY_LBP, MODALITY_GEOMETRY, MODALITY_ARCFACE]


class FeatureFusion:
    """
    Fuses multiple face feature modalities into a single descriptor vector.

    The modalities parameter controls which features are included, enabling
    ablation studies (testing all 10 combinations of 4 modalities).

    Parameters
    ----------
    modalities : list[str]
        Subset of ['hog', 'lbp', 'geometry', 'arcface'] to include.
        Order matters — sets the order of concatenation.
    weights : dict[str, float] or None
        Optional per-modality scaling weights applied AFTER L2-normalization.
        None = all weights = 1.0 (equal contribution).
        Example: {'arcface': 2.0} doubles ArcFace's contribution.
    """

    def __init__(
        self,
        modalities: list[str] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self.modalities = modalities or ALL_MODALITIES
        # Default: equal weights for all modalities (no prior preference)
        self.weights = weights or {m: 1.0 for m in self.modalities}

        # Validate all requested modalities are known
        for m in self.modalities:
            if m not in ALL_MODALITIES:
                raise ValueError(f"Unknown modality: '{m}'. Choose from {ALL_MODALITIES}")

    def fuse(
        self,
        hog: Optional[np.ndarray] = None,
        lbp: Optional[np.ndarray] = None,
        geometry: Optional[np.ndarray] = None,
        arcface: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fuse selected feature modalities into one concatenated descriptor.

        Parameters
        ----------
        hog, lbp, geometry, arcface : np.ndarray or None
            Feature vectors from each extractor.
            Pass None for unused modalities (must match self.modalities config).

        Returns
        -------
        np.ndarray — L2-normalized fused feature vector, shape (fused_dim,)

        Raises
        ------
        ValueError if a modality listed in self.modalities has None feature.
        """
        feature_map = {
            MODALITY_HOG:      hog,
            MODALITY_LBP:      lbp,
            MODALITY_GEOMETRY: geometry,
            MODALITY_ARCFACE:  arcface,
        }

        parts = []
        for modality in self.modalities:
            feat = feature_map[modality]
            if feat is None:
                raise ValueError(
                    f"Modality '{modality}' is enabled but its feature vector is None. "
                    "Check that the corresponding extractor is initialized in the pipeline."
                )

            # Step 1: L2-normalize — makes each modality unit-length regardless of dimension
            #         This is the key step that prevents high-D features from dominating
            normalized = self._l2_normalize(feat.ravel().astype(np.float32))

            # Step 2: Apply optional per-modality weight (default 1.0 = no change)
            weight = self.weights.get(modality, 1.0)
            parts.append(normalized * weight)

        # Step 3: Concatenate all modality vectors into one descriptor
        return np.concatenate(parts)

    def fuse_from_dict(self, feature_dict: dict) -> np.ndarray:
        """
        Convenience wrapper: fuse features supplied as a dictionary.

        WHY THIS METHOD: The pipeline's _extract_from_aligned() builds a dict
        of features keyed by modality name. This avoids having to unpack the
        dict manually at each call site.

        Parameters
        ----------
        feature_dict : dict with keys from ALL_MODALITIES
                       (missing keys are treated as None — fine if not in self.modalities)
        """
        return self.fuse(
            hog=feature_dict.get(MODALITY_HOG),
            lbp=feature_dict.get(MODALITY_LBP),
            geometry=feature_dict.get(MODALITY_GEOMETRY),
            arcface=feature_dict.get(MODALITY_ARCFACE),
        )

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """
        L2-normalize a vector to unit length.

        WHY GUARD AGAINST ZERO NORM: If a feature extractor fails completely
        (e.g., dlib can't find landmarks) it returns a zero vector.
        Dividing by zero would produce NaN, which propagates through PCA and SVM.
        Returning the zero vector as-is is safer — it signals "no information"
        without corrupting the rest of the feature vector.
        """
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v  # Return zero vector unchanged (feature extractor likely failed)
        return v / norm

    def compute_fused_dim(self, dims: dict[str, int]) -> int:
        """
        Compute the expected output dimension given per-modality dimensions.

        Useful for pre-allocating arrays or validating feature sizes before training.

        Parameters
        ----------
        dims : dict mapping modality name → feature vector length
               e.g. {'hog': 3780, 'lbp': 512, 'geometry': 159, 'arcface': 512}
        """
        return sum(dims[m] for m in self.modalities if m in dims)
