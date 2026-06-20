"""
reducer.py
----------
PCA dimensionality reduction module.
Reduces the high-dimensional fused feature vector to a manageable
size for SVM training.

Key considerations:
  - PCA retains n components explaining a target % of variance.
  - Fit only on training data; transform both train and test.
  - Whitening option available for SVM with RBF kernel.

Face Recognition on Skewed UAV Images
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAReducer:
    """
    Wraps sklearn PCA with:
      - Automatic component selection by variance threshold
      - StandardScaler preprocessing (zero mean, unit variance)
      - Save/load for deployment

    Parameters
    ----------
    n_components : int or float
        If int: number of components.
        If float (0 < x < 1): variance retention threshold (e.g., 0.95).
    whiten : bool
        Whiten PCA components (recommended for SVM with RBF kernel).
    scale_before_pca : bool
        Apply StandardScaler before PCA (recommended).
    """

    def __init__(
        self,
        n_components: int | float = 0.95,
        whiten: bool = True,
        scale_before_pca: bool = True,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.scale_before_pca = scale_before_pca

        self._scaler = StandardScaler() if scale_before_pca else None
        self._pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """
        Fit scaler and PCA on training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        if self._scaler is not None:
            X = self._scaler.fit_transform(X)
        self._pca.fit(X)
        self._is_fitted = True

        retained = self._pca.explained_variance_ratio_.sum()
        print(
            f"[PCAReducer] Fitted: {X.shape[1]} → {self._pca.n_components_} components "
            f"({retained*100:.1f}% variance retained)"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform feature vectors to PCA space.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        np.ndarray of shape (n_samples, n_components) or (n_components,)
        """
        if not self._is_fitted:
            raise RuntimeError("PCAReducer must be fitted before transform. Call fit() first.")

        squeeze = X.ndim == 1
        if squeeze:
            X = X[np.newaxis, :]

        if self._scaler is not None:
            X = self._scaler.transform(X)
        X_pca = self._pca.transform(X)

        return X_pca[0] if squeeze else X_pca

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step (for training set)."""
        self.fit(X)
        return self.transform(X)

    @property
    def n_components_(self) -> int:
        """Number of PCA components after fitting."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted yet.")
        return self._pca.n_components_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self._pca.explained_variance_ratio_

    def variance_curve(self) -> np.ndarray:
        """Cumulative explained variance for each component (for plotting)."""
        return np.cumsum(self._pca.explained_variance_ratio_)

    def save(self, path: str | Path) -> None:
        """Save fitted reducer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self._scaler, "pca": self._pca}, path)
        print(f"[PCAReducer] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PCAReducer":
        """Load a saved reducer from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._scaler = data["scaler"]
        obj._pca = data["pca"]
        obj._is_fitted = True
        obj.n_components = obj._pca.n_components_
        obj.whiten = obj._pca.whiten
        obj.scale_before_pca = obj._scaler is not None
        print(f"[PCAReducer] Loaded from {path} ({obj._pca.n_components_} components)")
        return obj

