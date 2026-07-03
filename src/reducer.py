"""
reducer.py
----------
WHAT : PCA-based dimensionality reduction with StandardScaler preprocessing.
WHY  : After fusion, the feature vector can be very large (e.g., 4963-D for all
       4 modalities). Training an SVM on such high-dimensional data is:
         1. SLOW — SVM complexity scales quadratically to cubically with features
         2. OVERFIT-PRONE — too many features relative to small gallery size (~30–100 samples)
         3. DOMINATED BY NOISE — many HOG/LBP dimensions carry little identity information

       PCA solves all three problems:
         - Reduces dimension to only the axes that explain 95% of the variance
         - Removes noise dimensions (low-variance principal components)
         - Creates uncorrelated features (better for SVM's RBF kernel)

       IMPORTANT: PCA is FIT ONLY on gallery (training) data, then applied
       (transform) to both gallery and probe. Never fit on probe data — that
       would leak test information into the model.

       WHY STANDARDSCALER BEFORE PCA:
         PCA is sensitive to feature scale. Without scaling, dimensions with
         large absolute values (e.g., raw pixel positions in geometry features)
         would dominate the principal components. StandardScaler makes every
         dimension have zero mean and unit variance before PCA runs.

       WHY WHITENING (whiten=True):
         Whitening additionally makes PCA components have unit variance, which
         is recommended for SVM with RBF kernel because it makes the kernel's
         gamma parameter scale-invariant across components.

Pipeline position: fused features → [PCAReducer] → SVM classifier

Face Recognition on Skewed UAV Images
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAReducer:
    """
    Wraps sklearn PCA with StandardScaler preprocessing and save/load support.

    Parameters
    ----------
    n_components : int or float
        If int   → exact number of PCA components to keep.
        If float → variance retention threshold (e.g., 0.95 = keep components
                   that together explain 95% of total variance).
                   Using a float is preferred — it auto-adapts to dataset size.
    whiten : bool
        Scale PCA components to unit variance after projection.
        Recommended when using SVM with RBF kernel (default: True).
    scale_before_pca : bool
        Apply StandardScaler before PCA to normalize feature magnitudes.
        Strongly recommended (default: True).
    """

    def __init__(
        self,
        n_components: int | float = 0.95,  # Default: retain 95% of variance
        whiten: bool = True,
        scale_before_pca: bool = True,
    ):
        self.n_components     = n_components
        self.whiten           = whiten
        self.scale_before_pca = scale_before_pca

        # Both scaler and PCA are initialized here but not fitted yet
        self._scaler = StandardScaler() if scale_before_pca else None
        self._pca    = PCA(n_components=n_components, whiten=whiten, random_state=42)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """
        Fit the scaler and PCA on the training (gallery) feature matrix.

        CRITICAL: Only call fit() on training/gallery data. Never on probe data.
        PCA learns the principal directions from the gallery distribution;
        applying it to probe ensures the transformation is unbiased.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) — gallery features

        Returns
        -------
        self (for method chaining)
        """
        # Scale features to zero-mean, unit-variance before PCA
        # (must fit_transform here so scaler parameters are set for later transforms)
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)

        self._pca.fit(X)
        self._is_fitted = True

        # Report how much dimension reduction happened
        retained = self._pca.explained_variance_ratio_.sum()
        print(
            f"[PCAReducer] Fitted: {X.shape[1]} → {self._pca.n_components_} components "
            f"({retained * 100:.1f}% variance retained)"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project feature vectors into PCA space.

        Handles both single vectors (1-D) and batches (2-D) transparently.
        The scaler uses TRAINING statistics (mean/std) for transformation —
        no information from the current batch is used.

        Parameters
        ----------
        X : np.ndarray — shape (n_features,) for single sample, or
                         shape (n_samples, n_features) for batch

        Returns
        -------
        np.ndarray — same shape structure but with n_components dimensions
        """
        if not self._is_fitted:
            raise RuntimeError("PCAReducer must be fitted before transform. Call fit() first.")

        # Remember if input was 1-D so we can squeeze the output back
        squeeze = X.ndim == 1
        if squeeze:
            X = X[np.newaxis, :]  # Add batch dimension: (n_features,) → (1, n_features)

        if self._scaler is not None:
            X = self._scaler.transform(X)  # Apply training-time normalization

        X_pca = self._pca.transform(X)

        return X_pca[0] if squeeze else X_pca  # Restore original shape convention

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step (convenience for training set only)."""
        self.fit(X)
        return self.transform(X)

    @property
    def n_components_(self) -> int:
        """Number of PCA components selected after fitting."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted yet.")
        return self._pca.n_components_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Per-component fraction of total variance explained (for plotting)."""
        return self._pca.explained_variance_ratio_

    def variance_curve(self) -> np.ndarray:
        """
        Cumulative explained variance as a function of number of components.
        Used by visualizer.plot_pca_variance() to show the elbow curve.
        """
        return np.cumsum(self._pca.explained_variance_ratio_)

    def save(self, path: str | Path) -> None:
        """
        Serialize the fitted scaler + PCA to disk using joblib.

        WHY JOBLIB instead of pickle: joblib handles large numpy arrays more
        efficiently (memory-mapped serialization). It's the sklearn-recommended
        persistence format.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "pca": self._pca}, path)
        print(f"[PCAReducer] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PCAReducer":
        """
        Load a previously saved reducer from disk.

        Uses classmethod (not __init__) to reconstruct the object from a pickle
        without needing to pass constructor arguments again — the saved state
        contains everything needed.
        """
        data = joblib.load(path)
        obj  = cls.__new__(cls)  # Skip __init__ — we'll set state directly
        obj._scaler        = data["scaler"]
        obj._pca           = data["pca"]
        obj._is_fitted     = True
        obj.n_components   = obj._pca.n_components_
        obj.whiten         = obj._pca.whiten
        obj.scale_before_pca = obj._scaler is not None
        print(f"[PCAReducer] Loaded from {path} ({obj._pca.n_components_} components)")
        return obj
