"""
pipeline.py
-----------
End-to-end face recognition pipeline orchestrator.

Integrates:
  Detection → Alignment → HOG + LBP + Geometry + ArcFace
  → Fusion → PCA → SVM → Identity Prediction

Supports:
  - Training mode (fit on gallery)
  - Inference mode (identify probe image)
  - Feature extraction mode (extract without classification)
  - Ablation mode (disable specific modalities)

Face Recognition on Skewed UAV Images
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.detection import FaceDetector
from src.alignment import FaceAligner
from src.features.hog_features import HOGExtractor
from src.features.lbp_features import LBPExtractor
from src.features.geometry_features import GeometryExtractor
from src.features.arcface_features import ArcFaceExtractor
from src.fusion import FeatureFusion, ALL_MODALITIES
from src.reducer import PCAReducer
from src.classifier import SVMClassifier


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline from raw image to identity prediction.

    Parameters
    ----------
    modalities : list[str]
        Feature modalities to use. Subset of ['hog', 'lbp', 'geometry', 'arcface'].
        Controls ablation experiments.
    pca_variance : float
        Variance retention threshold for PCA (default 0.95).
    svm_kernel : str
        SVM kernel type ('rbf' or 'linear').
    predictor_path : str
        Path to dlib shape predictor .dat file.
    arcface_model : str
        InsightFace model name.
    use_grid_search : bool
        Run SVM hyperparameter grid search.
    """

    def __init__(
        self,
        modalities: list[str] = None,
        pca_variance: float = 0.95,
        svm_kernel: str = "rbf",
        predictor_path: str = "models/shape_predictor_68_face_landmarks.dat",
        arcface_model: str = "buffalo_l",
        use_grid_search: bool = False,
    ):
        self.modalities = modalities or ALL_MODALITIES

        # --- Initialize components ---
        self.detector  = FaceDetector(min_face_size=20, confidence_threshold=0.85)
        self.aligner   = FaceAligner()

        self.hog_ext  = HOGExtractor() if "hog"      in self.modalities else None
        self.lbp_ext  = LBPExtractor() if "lbp"      in self.modalities else None
        self.geo_ext  = GeometryExtractor(predictor_path) if "geometry" in self.modalities else None
        self.arc_ext  = ArcFaceExtractor(arcface_model)   if "arcface"  in self.modalities else None

        self.fusion    = FeatureFusion(modalities=self.modalities)
        self.reducer   = PCAReducer(n_components=pca_variance)
        self.classifier = SVMClassifier(kernel=svm_kernel, use_grid_search=use_grid_search)

        self._is_trained = False

    # -------------------------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------------------------

    def extract_features(self, image: np.ndarray) -> np.ndarray | None:
        """
        Extract and fuse features from a single raw image.

        Steps:
          1. Detect face (MTCNN)
          2. Align to 112×112
          3. Extract enabled modalities
          4. L2-normalize and fuse

        Returns
        -------
        np.ndarray — fused feature vector, or None if no face detected.
        """
        face = self._detect_and_align(image)
        if face is None:
            return None
        return self._extract_from_aligned(face)

    def _detect_and_align(self, image: np.ndarray) -> np.ndarray | None:
        """Detect largest face and align to 112×112."""
        detection = self.detector.detect_largest(image)
        if detection is None:
            return None
        return self.aligner.align_from_detection(image, detection)

    def _extract_from_aligned(self, face: np.ndarray) -> np.ndarray:
        """Extract all enabled modalities from an aligned 112×112 face."""
        features = {}
        if self.hog_ext:
            features["hog"] = self.hog_ext.extract(face)
        if self.lbp_ext:
            features["lbp"] = self.lbp_ext.extract(face)
        if self.geo_ext:
            features["geometry"] = self.geo_ext.extract(face)
        if self.arc_ext:
            features["arcface"] = self.arc_ext.extract(face)
        return self.fusion.fuse_from_dict(features)

    # -------------------------------------------------------------------------
    # DATASET LOADING
    # -------------------------------------------------------------------------

    def load_dataset(
        self,
        dataset_dir: str | Path,
        augmentor=None,
        max_per_identity: int = None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load images from a directory with structure:
            dataset_dir/
              identity_001/ image1.jpg image2.jpg ...
              identity_002/ ...

        Parameters
        ----------
        augmentor : UAVAugmentor or None
            Apply UAV degradation during loading.
        max_per_identity : int or None
            Cap samples per identity (for balanced datasets).

        Returns
        -------
        (X, y) — feature matrix and label list
        """
        dataset_dir = Path(dataset_dir)
        X, y = [], []

        identity_dirs = sorted([
            d for d in dataset_dir.iterdir() if d.is_dir()
        ])

        if verbose:
            print(f"[Pipeline] Loading dataset from {dataset_dir} ({len(identity_dirs)} identities)...")

        for id_dir in tqdm(identity_dirs, disable=not verbose):
            identity = id_dir.name
            image_files = list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.png")) + \
                          list(id_dir.glob("*.jpeg")) + list(id_dir.glob("*.bmp"))

            if max_per_identity:
                image_files = image_files[:max_per_identity]

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Apply augmentation if provided
                if augmentor is not None:
                    img = augmentor.augment(img)

                feat = self.extract_features(img)
                if feat is not None:
                    X.append(feat)
                    y.append(identity)

        X = np.array(X, dtype=np.float32)
        if verbose:
            print(f"[Pipeline] Loaded {len(y)} samples from {len(set(y))} identities. "
                  f"Feature dim: {X.shape[1] if len(X) > 0 else 'N/A'}")
        return X, y

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: list[str],
    ) -> "FaceRecognitionPipeline":
        """
        Fit PCA reducer and SVM classifier on training features.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, raw_feature_dim)
        y_train : list of identity label strings

        Returns
        -------
        self
        """
        print(f"[Pipeline] Fitting PCA on {X_train.shape[0]} samples, "
              f"raw feature dim={X_train.shape[1]}...")
        X_pca = self.reducer.fit_transform(X_train)

        print(f"[Pipeline] Training SVM on {X_pca.shape[0]} samples, "
              f"PCA dim={X_pca.shape[1]}...")
        self.classifier.fit(X_pca, y_train)

        self._is_trained = True
        return self

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------

    def identify(
        self,
        image: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[str], list[float]] | None:
        """
        Identify person in an image.

        Parameters
        ----------
        image : np.ndarray — BGR image
        top_k : int — return top-k candidates

        Returns
        -------
        (labels, confidences) — top-k identity strings and confidence scores,
        or None if no face detected.
        """
        if not self._is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        feat = self.extract_features(image)
        if feat is None:
            return None

        feat_pca = self.reducer.transform(feat)
        top_labels, top_proba = self.classifier.predict_top_k(
            feat_pca[np.newaxis, :], k=top_k
        )
        return list(top_labels[0]), list(top_proba[0])

    def identify_from_aligned(
        self,
        face: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[str], list[float]]:
        """Identify from a pre-aligned 112×112 face (skips detection step)."""
        if not self._is_trained:
            raise RuntimeError("Pipeline not trained.")

        feat = self._extract_from_aligned(face)
        feat_pca = self.reducer.transform(feat)
        top_labels, top_proba = self.classifier.predict_top_k(
            feat_pca[np.newaxis, :], k=top_k
        )
        return list(top_labels[0]), list(top_proba[0])

    # -------------------------------------------------------------------------
    # EVALUATION HELPERS
    # -------------------------------------------------------------------------

    def extract_features_batch(
        self,
        dataset_dir: str | Path,
        augmentor=None,
        max_per_identity: int = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load dataset and return PCA-transformed features for evaluation."""
        X, y = self.load_dataset(dataset_dir, augmentor, max_per_identity)
        if len(X) == 0:
            return np.array([]), []
        return self.reducer.transform(X), y

    # -------------------------------------------------------------------------
    # SAVE / LOAD
    # -------------------------------------------------------------------------

    def save(self, model_dir: str | Path = "models") -> None:
        """Save PCA and SVM models to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.reducer.save(model_dir / "pca_reducer.pkl")
        self.classifier.save(model_dir / "svm_classifier.pkl")
        print(f"[Pipeline] Models saved to {model_dir}/")

    def load(self, model_dir: str | Path = "models") -> "FaceRecognitionPipeline":
        """Load PCA and SVM models from disk."""
        model_dir = Path(model_dir)
        self.reducer = PCAReducer.load(model_dir / "pca_reducer.pkl")
        self.classifier = SVMClassifier.load(model_dir / "svm_classifier.pkl")
        self._is_trained = True
        return self

