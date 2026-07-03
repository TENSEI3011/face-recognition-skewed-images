"""
classifier.py
-------------
WHAT : SVM-based face identity classifier with probability calibration.
WHY SVM: Support Vector Machines are the go-to classifier for face recognition
       with small training sets (few images per identity). Reasons:
         1. SVM finds the maximum-margin hyperplane — more robust to overfitting
            than neural classifiers on <100 training samples per class.
         2. RBF kernel handles non-linear, curved decision boundaries between
            identity clusters in PCA space.
         3. 'class_weight=balanced' corrects for unequal numbers of gallery
            images per person (common in real datasets).
         4. Works well on the low-dimensional PCA output (typically 20–80 components).

       WHY NOT KNN: k-NN requires storing and searching all gallery embeddings
       at inference time. SVM compresses the decision boundary into support
       vectors, making inference faster.

       WHY NOT NEURAL CLASSIFIER: We have too few gallery samples per identity
       (~3–20) to fine-tune a neural network. SVM generalises far better at
       this scale.

       PROBABILITY CALIBRATION (Platt Scaling):
         Standard SVM (decision_function) produces raw margin scores, not
         probabilities. Platt scaling fits a sigmoid on top of the margin scores
         using cross-validation, giving calibrated probabilities.
         WHY NEEDED: The evaluation framework (Rank-k, EER) requires confidence
         scores for ranking. Without calibrated probabilities, we cannot compute
         TAR@FAR or EER meaningfully.

Pipeline position: PCA features → [SVMClassifier] → ranked identity predictions

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
      - RBF or Linear kernel (selectable)
      - Platt scaling for calibrated probability outputs
      - Optional GridSearchCV hyperparameter tuning
      - LabelEncoder for string → integer → string label mapping
      - Save/load for deployment with demo.py

    Parameters
    ----------
    kernel : str
        'rbf' (default) — best for curved cluster boundaries in PCA space.
        'linear' — faster, use if n_components > 200 (rarely needed).
    C : float
        SVM regularization. Higher C = smaller margin, more training points
        correctly classified (risk of overfitting). Lower C = wider margin,
        more robust but may misclassify training samples.
        Default 10.0 is a reasonable start; GridSearchCV tunes this.
    gamma : str or float
        RBF kernel bandwidth. 'scale' = 1/(n_features × X.var()) — adapts
        to feature scale automatically and is usually optimal without tuning.
    use_grid_search : bool
        Run GridSearchCV over [C, gamma] combinations to find the best settings.
        Adds training time (~minutes) but often improves accuracy.
    probability : bool
        Enable Platt scaling for calibrated confidence scores (required for
        Rank-k evaluation and EER computation).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 10.0,
        gamma: str | float = "scale",
        use_grid_search: bool = False,
        probability: bool = True,
    ):
        self.kernel          = kernel
        self.C               = C
        self.gamma           = gamma
        self.use_grid_search = use_grid_search
        self.probability     = probability

        self._label_encoder = LabelEncoder()  # Maps string labels ↔ integers
        self._svm           = None
        self._is_fitted     = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: list | np.ndarray,
    ) -> "SVMClassifier":
        """
        Train the SVM on PCA-reduced gallery features.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_components)
                  PCA-transformed gallery features
        y_train : array-like of string identity labels (e.g., ['alice', 'bob', ...])

        Returns
        -------
        self (for method chaining)
        """
        # Convert string labels ('alice', 'bob') to integers (0, 1, ...) for sklearn
        # The encoder remembers the mapping so we can decode predictions back to strings
        y_enc     = self._label_encoder.fit_transform(y_train)
        n_classes = len(self._label_encoder.classes_)
        print(f"[SVMClassifier] Training on {X_train.shape[0]} samples, {n_classes} identities.")

        if self.use_grid_search:
            # Run cross-validated grid search to find optimal C and gamma
            svm = self._grid_search(X_train, y_enc)
        else:
            svm = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=self.probability,
                class_weight="balanced",  # Corrects for unequal gallery sizes per identity
                random_state=42,
            )

        if self.probability and not self.use_grid_search:
            # Platt scaling requires cross-validation to fit the sigmoid
            # WHY cv=2 MINIMUM: With small galleries (e.g., 3 images/identity),
            # we need at least 2 samples per class per fold. We compute the
            # maximum feasible cv value to avoid empty fold errors.
            min_class_count = min(np.bincount(y_enc))
            cv = min(3, min_class_count // 2)
            cv = max(2, cv)  # Ensure at least 2 folds (1 fold = no validation)
            self._svm = CalibratedClassifierCV(svm, method="sigmoid", cv=cv)
        else:
            # GridSearchCV already handles probability internally, or no calibration needed
            self._svm = svm

        self._svm.fit(X_train, y_enc)
        self._is_fitted = True
        print(f"[SVMClassifier] Training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely identity label for each sample.

        Returns the TOP-1 prediction only (no ranking). Use predict_top_k
        for Rank-k evaluation.

        Returns
        -------
        np.ndarray of string labels, shape (n_samples,)
        """
        self._check_fitted()
        # Handle single-sample input (1-D array) by adding batch dimension
        y_enc = self._svm.predict(X if X.ndim == 2 else X[np.newaxis, :])
        return self._label_encoder.inverse_transform(y_enc)  # Decode int → string

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict ALL identity labels ranked by confidence score.

        Returns labels sorted from most to least likely, enabling Rank-k evaluation
        (check if correct identity is in top-k predictions).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_components) or (n_components,)

        Returns
        -------
        labels : np.ndarray, shape (n_samples, n_classes) — labels sorted by confidence
        proba  : np.ndarray, shape (n_samples, n_classes) — corresponding probabilities
        """
        self._check_fitted()

        if X.ndim == 1:
            X = X[np.newaxis, :]  # Add batch dimension for single sample

        if hasattr(self._svm, "predict_proba"):
            # Standard path: Platt-calibrated probabilities (recommended)
            proba = self._svm.predict_proba(X)
        else:
            # Fallback: if SVM lacks probability support, use softmax of decision scores
            # This is less calibrated but still gives a meaningful ranking
            scores = self._svm.decision_function(X)
            proba  = self._softmax(scores)

        # Sort each row in descending probability order (highest confidence first)
        sorted_idx    = np.argsort(-proba, axis=1)
        sorted_labels = np.array([
            self._label_encoder.classes_[idx] for idx in sorted_idx
        ])
        sorted_proba  = np.take_along_axis(proba, sorted_idx, axis=1)

        return sorted_labels, sorted_proba

    def predict_top_k(
        self,
        X: np.ndarray,
        k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the top-k most likely identity labels per sample.

        Used for Rank-k CMC evaluation: Rank-1 = top-1 correct;
        Rank-5 = correct identity appears within top-5 predictions.

        Returns
        -------
        top_k_labels : np.ndarray, shape (n_samples, k)
        top_k_proba  : np.ndarray, shape (n_samples, k)
        """
        labels, proba = self.predict_proba(X)
        return labels[:, :k], proba[:, :k]

    @property
    def classes_(self) -> np.ndarray:
        """Original string identity labels (as seen by the label encoder)."""
        return self._label_encoder.classes_

    def save(self, path: str | Path) -> None:
        """
        Serialize the fitted SVM + label encoder to disk.

        Both are saved together because the encoder is needed to decode
        integer predictions back to string identity names during inference.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"svm": self._svm, "encoder": self._label_encoder}, path)
        print(f"[SVMClassifier] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SVMClassifier":
        """Load a previously saved classifier from disk."""
        data = joblib.load(path)
        obj  = cls.__new__(cls)  # Bypass __init__ to avoid re-initializing
        obj._svm           = data["svm"]
        obj._label_encoder = data["encoder"]
        obj._is_fitted     = True
        print(f"[SVMClassifier] Loaded from {path} ({len(obj._label_encoder.classes_)} identities)")
        return obj

    def _check_fitted(self):
        """Raise a clear error if predict is called before fit."""
        if not self._is_fitted:
            raise RuntimeError("SVMClassifier must be fitted before prediction. Call fit() first.")

    def _grid_search(self, X: np.ndarray, y: np.ndarray) -> SVC:
        """
        Run GridSearchCV to find optimal C and gamma hyperparameters.

        WHY STRATIFIED K-FOLD: Ensures each fold has representatives from all
        identity classes, which is critical with small datasets where one fold
        might otherwise miss an identity entirely.

        WHY THESE PARAM RANGES:
          C: [0.1, 1, 10, 100] — spans 3 orders of magnitude around the default (10)
          gamma: ['scale', 'auto', 0.001, 0.01] — 'scale' usually wins, but explicit
                 values help when the feature variance is unusual.
        """
        print("[SVMClassifier] Running GridSearchCV (this may take a few minutes)...")
        param_grid = {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01],
        }
        base = SVC(
            kernel=self.kernel,
            probability=self.probability,
            class_weight="balanced",
            random_state=42,
        )
        # n_splits=3: 3-fold CV — gives a reliable estimate without too long training time
        cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(base, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X, y)
        print(
            f"[SVMClassifier] Best params: {grid.best_params_} | "
            f"CV accuracy: {grid.best_score_:.4f}"
        )
        return grid.best_estimator_

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax — used as fallback when predict_proba is unavailable.

        WHY SUBTRACT MAX: Prevents overflow in exp(large numbers).
        Mathematically equivalent to standard softmax.
        """
        x = x - x.max(axis=1, keepdims=True)  # Stability: subtract row-wise max
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)
