"""
pipeline.py
-----------
WHAT : End-to-end face recognition pipeline orchestrator.
WHY  : This single class wires all components together in the correct order,
       so callers (experiment scripts, demo.py) don't need to know about
       individual extractors or the specific sequence of operations.

       Full pipeline flow:
         RAW IMAGE
           ↓ FaceDetector (MTCNN) — locate face, get eye landmarks
           ↓ FaceAligner         — affine warp to 112×112 canonical crop
           ↓ Feature Extractors  — HOG + LBP + Geometry + ArcFace (configurable)
           ↓ FeatureFusion       — L2-normalize each modality, then concatenate
           ↓ PCAReducer          — reduce to 20–80 principal components
           ↓ SVMClassifier       — predict ranked identity labels
           ↓ IDENTITY + CONFIDENCE

       WHY ONE CLASS: Encapsulating the whole pipeline in one object makes it
       easy to save/load, swap out components for ablation studies (e.g.,
       disable ArcFace), and run inference in demo.py without duplicating logic.

Face Recognition on Skewed UAV Images
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm  # Progress bars for long dataset loading operations

from src.detection  import FaceDetector
from src.alignment  import FaceAligner
from src.features.hog_features      import HOGExtractor
from src.features.lbp_features      import LBPExtractor
from src.features.geometry_features import GeometryExtractor
from src.features.arcface_features  import ArcFaceExtractor
from src.fusion     import FeatureFusion, ALL_MODALITIES
from src.reducer    import PCAReducer
from src.classifier import SVMClassifier


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline from raw image to identity prediction.

    ABLATION SUPPORT: The 'modalities' parameter allows running with any
    subset of the 4 feature types. This powers run_ablation.py which tests
    all 10 possible combinations (2^4 - 6 invalid = 10 valid non-empty subsets).

    Parameters
    ----------
    modalities : list[str]
        Feature modalities to activate. Subset of ['hog', 'lbp', 'geometry', 'arcface'].
        Unselected modalities are not initialized (saves memory and compute time).
    pca_variance : float
        Fraction of variance PCA should retain (e.g., 0.95 = 95%).
    svm_kernel : str
        'rbf' or 'linear'. RBF is better for non-linear identity boundaries.
    predictor_path : str
        Path to the dlib 68-point landmark .dat file (needed for geometry features).
    arcface_model : str
        InsightFace model name. 'buffalo_l' = highest accuracy.
    use_grid_search : bool
        Whether to run GridSearchCV for SVM hyperparameter tuning.
    """

    def __init__(
        self,
        modalities:    list[str] = None,
        pca_variance:  float = 0.95,
        svm_kernel:    str = "rbf",
        predictor_path: str = "models/shape_predictor_68_face_landmarks.dat",
        arcface_model: str = "buffalo_l",
        use_grid_search: bool = False,
    ):
        self.modalities = modalities or ALL_MODALITIES

        # ── Detection & Alignment (always active regardless of modalities) ────
        # min_face_size=20: catch UAV faces as small as 20px (30m altitude)
        # confidence_threshold=0.85: balance between catching real faces
        #                            and filtering false positives
        self.detector = FaceDetector(min_face_size=20, confidence_threshold=0.5)
        self.aligner  = FaceAligner()

        # ── Feature Extractors (only instantiated if modality is requested) ───
        # WHY CONDITIONAL INIT: Loading ArcFace (300MB model) when it's not needed
        # would waste memory. Only geometry needs a file path; others are parameter-only.
        self.hog_ext = HOGExtractor()                      if "hog"      in self.modalities else None
        self.lbp_ext = LBPExtractor()                      if "lbp"      in self.modalities else None
        self.geo_ext = GeometryExtractor(predictor_path)   if "geometry" in self.modalities else None
        self.arc_ext = ArcFaceExtractor(arcface_model)     if "arcface"  in self.modalities else None

        # ── Downstream Components ─────────────────────────────────────────────
        self.fusion     = FeatureFusion(modalities=self.modalities)
        self.reducer    = PCAReducer(n_components=pca_variance)
        self.classifier = SVMClassifier(kernel=svm_kernel, use_grid_search=use_grid_search)

        self._is_trained = False  # Guards against calling identify() before train()

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features(self, image: np.ndarray) -> np.ndarray | None:
        """
        Extract fused features from a SINGLE raw image (full pipeline).

        Steps:
          1. Detect the largest face with MTCNN
          2. Align to 112×112 using eye-corner landmarks
          3. Extract enabled modalities (HOG / LBP / Geometry / ArcFace)
          4. L2-normalize each modality and concatenate

        Returns
        -------
        np.ndarray — fused feature vector, or None if no face was detected.
        None causes the calling code to skip this image gracefully.
        """
        face = self._detect_and_align(image)
        if face is None:
            return None  # No face found — skip this image
        return self._extract_from_aligned(face)

    def _detect_and_align(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detect the largest face in the image and align it to 112×112.
        Returns aligned BGR face crop or None if no face found.
        """
        detection = self.detector.detect_largest(image)
        if detection is None:
            return None
        return self.aligner.align_from_detection(image, detection)

    def _extract_from_aligned(self, face: np.ndarray) -> np.ndarray:
        """
        Extract all enabled feature modalities from an already-aligned 112×112 face.
        Called directly in batch processing to skip re-detection.
        """
        features = {}

        # Each extractor is called only if it was initialised (modality is active)
        if self.hog_ext:
            features["hog"]      = self.hog_ext.extract(face)
        if self.lbp_ext:
            features["lbp"]      = self.lbp_ext.extract(face)
        if self.geo_ext:
            features["geometry"] = self.geo_ext.extract(face)
        if self.arc_ext:
            features["arcface"]  = self.arc_ext.extract(face)

        # Fuse: L2-normalize each modality vector, then concatenate all
        return self.fusion.fuse_from_dict(features)

    # ─────────────────────────────────────────────────────────────────────────
    # DATASET LOADING
    # ─────────────────────────────────────────────────────────────────────────

    def load_dataset(
        self,
        dataset_dir: str | Path,
        augmentor=None,
        max_per_identity: int = None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load a face recognition dataset organized as subdirectories per identity.

        Expected structure:
            dataset_dir/
              identity_001/ img1.jpg img2.jpg ...
              identity_002/ ...

        WHY THIS STRUCTURE: It's the standard format for gallery/probe datasets.
        Each subdirectory name becomes the identity label (y). No separate
        label file is needed.

        Parameters
        ----------
        augmentor : UAVAugmentor or None
            Apply UAV degradation to every image during loading.
            Used to simulate probe conditions in degradation experiments.
        max_per_identity : int or None
            Cap on images per identity. Useful to balance unequal class sizes
            and speed up experiments with large datasets.

        Returns
        -------
        X : np.ndarray of shape (n_samples, raw_feature_dim) — feature matrix
        y : list of str — identity labels (one per sample)
        """
        dataset_dir    = Path(dataset_dir)
        X, y           = [], []
        identity_dirs  = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

        if verbose:
            print(f"[Pipeline] Loading dataset from {dataset_dir} ({len(identity_dirs)} identities)...")

        for id_dir in tqdm(identity_dirs, disable=not verbose):
            identity    = id_dir.name  # Directory name = identity label
            image_files = (
                list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.png")) +
                list(id_dir.glob("*.jpeg")) + list(id_dir.glob("*.bmp"))
            )

            # Cap images per identity if requested
            if max_per_identity:
                image_files = image_files[:max_per_identity]

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue  # Skip corrupt or unreadable files

                # Optionally apply UAV degradation before feature extraction
                if augmentor is not None:
                    img = augmentor.augment(img)

                h, w = img.shape[:2]
                feat = None

                # Fast path: small images (<=150px) are pre-aligned face crops.
                # Skip SCRFD detection entirely — it always fails on tight crops
                # (no background context) and wastes time upscaling to 640x640.
                if max(h, w) <= 150:
                    try:
                        aligned = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
                        feat = self._extract_from_aligned(aligned)
                    except Exception:
                        feat = None
                else:
                    feat = self.extract_features(img)
                    # Secondary fallback: larger images where detection failed
                    if feat is None and max(h, w) <= 200:
                        try:
                            aligned = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
                            feat = self._extract_from_aligned(aligned)
                        except Exception:
                            feat = None

                if feat is not None:
                    X.append(feat)
                    y.append(identity)

        X = np.array(X, dtype=np.float32)
        if verbose:
            print(
                f"[Pipeline] Loaded {len(y)} samples from {len(set(y))} identities. "
                f"Feature dim: {X.shape[1] if len(X) > 0 else 'N/A'}"
            )
        return X, y

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: list[str],
    ) -> "FaceRecognitionPipeline":
        """
        Fit PCA reducer and SVM classifier on gallery features.

        IMPORTANT: This must be called AFTER load_dataset() on gallery data.
        PCA and SVM are fitted on the gallery (training) set ONLY.
        Probe (test) features are transformed through the fitted PCA but
        never used to fit it — this prevents data leakage.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, raw_feature_dim) — gallery features
        y_train : list[str] — identity labels for gallery samples
        """
        print(
            f"[Pipeline] Fitting PCA on {X_train.shape[0]} samples, "
            f"raw feature dim={X_train.shape[1]}..."
        )
        # fit_transform: fit PCA on gallery, then project gallery into PCA space
        X_pca = self.reducer.fit_transform(X_train)

        print(
            f"[Pipeline] Training SVM on {X_pca.shape[0]} samples, "
            f"PCA dim={X_pca.shape[1]}..."
        )
        self.classifier.fit(X_pca, y_train)

        self._is_trained = True
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ─────────────────────────────────────────────────────────────────────────

    def identify(
        self,
        image: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[str], list[float]] | None:
        """
        Identify the person in a raw BGR image.

        Full inference path:
          image → detect → align → extract features → PCA transform → SVM predict

        Parameters
        ----------
        image : np.ndarray — full BGR frame (any size)
        top_k : int — number of ranked candidates to return

        Returns
        -------
        (labels, confidences) — top-k identity strings and their confidence scores,
        or None if no face was detected in the image.
        """
        if not self._is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        feat = self.extract_features(image)
        if feat is None:
            return None  # No face detected

        # Project raw features into PCA space (uses training-time scaler/PCA params)
        feat_pca = self.reducer.transform(feat)

        # Get top-k ranked predictions with confidence scores
        top_labels, top_proba = self.classifier.predict_top_k(
            feat_pca[np.newaxis, :], k=top_k  # Add batch dimension
        )
        return list(top_labels[0]), list(top_proba[0])

    def identify_from_aligned(
        self,
        face: np.ndarray,
        top_k: int = 5,
    ) -> tuple[list[str], list[float]]:
        """
        Identify from a PRE-ALIGNED 112×112 face crop (skips detection step).

        WHY THIS METHOD: In demo.py, we already have a cropped face from the
        detection step. Re-running detection on a crop wastes time.
        This shortcut skips straight to feature extraction.
        """
        if not self._is_trained:
            raise RuntimeError("Pipeline not trained. Call train() first.")

        feat     = self._extract_from_aligned(face)
        feat_pca = self.reducer.transform(feat)
        top_labels, top_proba = self.classifier.predict_top_k(
            feat_pca[np.newaxis, :], k=top_k
        )
        return list(top_labels[0]), list(top_proba[0])

    # ─────────────────────────────────────────────────────────────────────────
    # EVALUATION HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features_batch(
        self,
        dataset_dir: str | Path,
        augmentor=None,
        max_per_identity: int = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Load a dataset and return PCA-projected features (for evaluation).

        WHY THIS WRAPPER: Experiment scripts need PCA-projected features to
        compute metrics (CMC, EER, etc.). This combines load_dataset() +
        reducer.transform() in one convenient call.
        """
        X, y = self.load_dataset(dataset_dir, augmentor, max_per_identity)
        if len(X) == 0:
            return np.array([]), []
        return self.reducer.transform(X), y

    # ─────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, model_dir: str | Path = "models") -> None:
        """
        Save the fitted PCA reducer, SVM classifier, and modalities metadata to disk.

        WHY SAVE MODALITIES: demo.py needs to reconstruct the pipeline with the
        same modalities that were used during training. Without this file,
        demo.py wouldn't know which feature extractors to initialize.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.reducer.save(model_dir / "pca_reducer.pkl")
        self.classifier.save(model_dir / "svm_classifier.pkl")

        # Save the active modalities alongside the model weights
        meta = {"modalities": self.modalities}
        with open(model_dir / "modalities.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Pipeline] Models saved to {model_dir}/  (modalities: {self.modalities})")

    def load(self, model_dir: str | Path = "models") -> "FaceRecognitionPipeline":
        """Load PCA and SVM from disk (modalities must already be set in constructor)."""
        model_dir = Path(model_dir)
        self.reducer    = PCAReducer.load(model_dir / "pca_reducer.pkl")
        self.classifier = SVMClassifier.load(model_dir / "svm_classifier.pkl")
        self._is_trained = True
        return self

    @staticmethod
    def load_modalities(model_dir: str | Path) -> list[str]:
        """
        Read the modalities list saved alongside a trained model.

        WHY STATIC METHOD: demo.py needs the modalities before it can construct
        the pipeline object (to know which extractors to initialize). This method
        reads the JSON without requiring an existing pipeline instance.

        Returns 'arcface' as a safe default if the file doesn't exist
        (handles models saved before modalities.json was added).
        """
        meta_path = Path(model_dir) / "modalities.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f).get("modalities", ["arcface"])
        return ["arcface"]  # Safe default — ArcFace-only model is the most common
