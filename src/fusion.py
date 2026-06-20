"""
fusion.py
---------
Feature fusion module.
Concatenates HOG, LBP, Geometry, and ArcFace features into a single
descriptor after per-modality L2 normalization.

Normalization is applied per modality before concatenation to prevent
high-magnitude features (e.g., ArcFace 512-D) from dominating
low-magnitude ones (e.g., geometry ratios).

Face Recognition on Skewed UAV Images
"""

import numpy as np
from typing import Optional


# Modality identifiers (used for ablation studies)
MODALITY_HOG      = "hog"
MODALITY_LBP      = "lbp"
MODALITY_GEOMETRY = "geometry"
MODALITY_ARCFACE  = "arcface"

ALL_MODALITIES = [MODALITY_HOG, MODALITY_LBP, MODALITY_GEOMETRY, MODALITY_ARCFACE]


class FeatureFusion:
    """
    Fuses multiple face feature modalities into a single descriptor.

    Fusion strategy:
      1. L2-normalize each modality vector independently.
      2. Concatenate into one vector.
      3. Optional: apply a per-modality weight (default: equal weights).

    Parameters
    ----------
    modalities : list[str]
        Subset of ['hog', 'lbp', 'geometry', 'arcface'] to include.
        Use this to run ablation studies.
    weights : dict[str, float] or None
        Per-modality weights. None = equal weights (all 1.0).
    """

    def __init__(
        self,
        modalities: list[str] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self.modalities = modalities or ALL_MODALITIES
        self.weights = weights or {m: 1.0 for m in self.modalities}

        # Validate
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
        Fuse selected feature modalities.

        Parameters
        ----------
        hog, lbp, geometry, arcface : np.ndarray or None
            Feature vectors from each extractor. Pass None if not used.

        Returns
        -------
        np.ndarray — concatenated, normalized fused descriptor.

        Raises
        ------
        ValueError if a required modality's feature is None.
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
                    f"Modality '{modality}' is selected but feature is None."
                )
            # L2 normalize each modality
            normalized = self._l2_normalize(feat.ravel().astype(np.float32))
            # Apply weight
            weight = self.weights.get(modality, 1.0)
            parts.append(normalized * weight)

        return np.concatenate(parts)

    def fuse_from_dict(self, feature_dict: dict) -> np.ndarray:
        """
        Fuse features supplied as a dictionary.

        Parameters
        ----------
        feature_dict : dict with keys from ALL_MODALITIES
        """
        return self.fuse(
            hog=feature_dict.get(MODALITY_HOG),
            lbp=feature_dict.get(MODALITY_LBP),
            geometry=feature_dict.get(MODALITY_GEOMETRY),
            arcface=feature_dict.get(MODALITY_ARCFACE),
        )

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """L2-normalize a vector. Returns zero vector if norm is near zero."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return v / norm

    def compute_fused_dim(self, dims: dict[str, int]) -> int:
        """
        Compute the expected output dimension given per-modality dimensions.

        Parameters
        ----------
        dims : dict mapping modality name → feature dimension
        """
        return sum(dims[m] for m in self.modalities if m in dims)

