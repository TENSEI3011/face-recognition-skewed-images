"""
classifier.py
-------------
SVM-based face identity classifier.
Uses RBF kernel SVM (or linear for high-dimensional PCA output).

Design rationale:
  - SVM generalizes well in high-dimensional spaces with limited training data.
  - RBF kernel handles non-linear boundaries between identity clusters.
  - Probability calibration (Platt scaling) enables confidence scores.
  - GridSearchCV automates hyperparameter selection (C, gamma).

Face Recognition on Skewed UAV Images
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


class SVMClassifier:
    """
    SVM face identity classifier with:
      - RBF or Linear kernel
      - Probability calibration via Platt scaling
      - Optional hyperparameter search (GridSearchCV)
      - Label encoding for string identity labels
      - Save/load for deployment

    Parameters
    ----------
    kernel : str
        'rbf' (default) or 'linear'. Use 'linear' if n_components > 200.
    C : float
        SVM regularization parameter. Overridden if use_grid_search=True.
    gamma : str or float
        RBF kernel coefficient. 'scale' = 1/(n_features * X.var()).
    use_grid_search : bool
        Run GridSearchCV to find optimal C and gamma.
    probability : bool
        If True, enable probability estimates (required for Rank-k evaluation).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 10.0,
        gamma: str | float = "scale",
        use_grid_search: bool = False,
        probability: bool = True,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.use_grid_search = use_grid_search
        self.probability = probability

        self._label_encoder = LabelEncoder()
        self._svm = None
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: list | np.ndarray,
    ) -> "SVMClassifier":
        """
        Train SVM on feature matrix.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_components)
        y_train : array-like of identity labels (strings or ints)

        Returns
        -------
        self
        """
        # Encode string labels to integers
        y_enc = self._label_encoder.fit_transform(y_train)
        n_classes = len(self._label_encoder.classes_)
        print(f"[SVMClassifier] Training on {X_train.shape[0]} samples, {n_classes} identities.")

        if self.use_grid_search:
            svm = self._grid_search(X_train, y_enc)
        else:
            svm = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=self.probability,
                class_weight="balanced",
                random_state=42,
            )

        # Calibrate probabilities with Platt scaling
        if self.probability and not self.use_grid_search:
            cv = min(5, min(np.bincount(y_enc)))
            cv = max(2, cv)
            self._svm = CalibratedClassifierCV(svm, method="sigmoid", cv=cv)
        else:
            self._svm = svm

        self._svm.fit(X_train, y_enc)
        self._is_fitted = True
        print(f"[SVMClassifier] Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict identity labels.

        Returns
        -------
        np.ndarray of string labels.
        """
        self._check_fitted()
        y_enc = self._svm.predict(X if X.ndim == 2 else X[np.newaxis, :])
        return self._label_encoder.inverse_transform(y_enc)

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict ranked identity labels with confidence scores.

        Returns
        -------
        labels     : np.ndarray of shape (n_samples, n_classes) — sorted by probability
        proba      : np.ndarray of shape (n_samples, n_classes) — corresponding probabilities
        """
        self._check_fitted()
        if X.ndim == 1:
            X = X[np.newaxis, :]

        if hasattr(self._svm, "predict_proba"):
            proba = self._svm.predict_proba(X)
        else:
            # Linear SVM without calibration: use decision function as proxy
            scores = self._svm.decision_function(X)
            proba = self._softmax(scores)

        # Sort each row by decreasing probability
        sorted_idx = np.argsort(-proba, axis=1)
        sorted_labels = np.array([
            self._label_encoder.classes_[idx] for idx in sorted_idx
        ])
        sorted_proba = np.take_along_axis(proba, sorted_idx, axis=1)

        return sorted_labels, sorted_proba

    def predict_top_k(
        self,
        X: np.ndarray,
        k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return top-k predictions per sample (for Rank-k evaluation).

        Returns
        -------
        top_k_labels : np.ndarray of shape (n_samples, k)
        top_k_proba  : np.ndarray of shape (n_samples, k)
        """
        labels, proba = self.predict_proba(X)
        return labels[:, :k], proba[:, :k]

    @property
    def classes_(self) -> np.ndarray:
        """Original string identity labels."""
        return self._label_encoder.classes_

    def save(self, path: str | Path) -> None:
        """Save fitted classifier to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"svm": self._svm, "encoder": self._label_encoder}, path)
        print(f"[SVMClassifier] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SVMClassifier":
        """Load a saved classifier from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._svm = data["svm"]
        obj._label_encoder = data["encoder"]
        obj._is_fitted = True
        print(f"[SVMClassifier] Loaded from {path} ({len(obj._label_encoder.classes_)} identities)")
        return obj

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("SVMClassifier must be fitted before prediction.")

    def _grid_search(self, X: np.ndarray, y: np.ndarray) -> SVC:
        """Run GridSearchCV for optimal C and gamma."""
        print("[SVMClassifier] Running GridSearchCV (this may take a few minutes)...")
        param_grid = {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01],
        }
        base = SVC(kernel=self.kernel, probability=self.probability,
                   class_weight="balanced", random_state=42)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(base, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X, y)
        print(f"[SVMClassifier] Best params: {grid.best_params_} | "
              f"CV accuracy: {grid.best_score_:.4f}")
        return grid.best_estimator_

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

