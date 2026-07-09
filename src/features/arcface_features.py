"""
arcface_features.py
-------------------
WHAT : ArcFace deep face embedding extractor using InsightFace (ONNX runtime).
WHY  : ArcFace is the state-of-the-art face recognition model as of 2024.
       It produces a 512-dimensional L2-normalized embedding where:
         - Same person → embeddings close together (high cosine similarity)
         - Different people → embeddings far apart
       The model was trained with Additive Angular Margin Loss, which creates
       more discriminative, tightly clustered identity representations than
       standard softmax or triplet-loss models.

       WHY INCLUDE ARCFACE IN A HYBRID PIPELINE:
         ArcFace alone performs very well at frontal faces but degrades at
         extreme UAV angles (>60° yaw, >45° pitch). Combining it with
         HOG + LBP + Geometry gives the pipeline fallback descriptors when
         ArcFace embeddings become unreliable.

       MODEL: 'buffalo_l' (InsightFace)
         - Backbone: ResNet-100
         - Training: MS1MV3 dataset (~5.8M images, 93K identities)
         - Output: 512-D L2-normalized embedding
         - ctx_id=0: uses GPU if available; ctx_id=-1 forces CPU
         WHY buffalo_l: The 'l' (large) model has the best accuracy among
         InsightFace's models. buffalo_s (small) trades accuracy for speed.

       FALLBACK STRATEGY:
         InsightFace runs its own internal face detector. If it can't find a
         face in the 112×112 aligned crop (very unusual), we pad the image to
         224×224 and retry. This handles cases where the face is extremely
         close to the edge of the crop.

Reference: Deng et al. (2019). ArcFace: Additive Angular Margin Loss for Deep
           Face Recognition. CVPR 2019.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from typing import Optional

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[ArcFaceExtractor] WARNING: insightface not installed. "
          "ArcFace features will return zero vectors. "
          "Install with: pip install insightface onnxruntime")


class ArcFaceExtractor:
    """
    Extracts 512-D ArcFace embeddings using InsightFace's pretrained buffalo_l model.

    Parameters
    ----------
    model_name : str
        InsightFace model pack name ('buffalo_l' = highest accuracy).
    ctx_id : int
        Device ID: 0 = first GPU (CUDA), -1 = CPU only.
        WHY DEFAULT 0: GPU inference is ~10× faster on ArcFace's ResNet-100.
        Falls back gracefully to CPU if no GPU is available.
    det_size : tuple
        Detection input resolution. 640×640 allows detecting faces as small
        as ~15px, which covers UAV faces at 30m+ altitude.
    """

    FEATURE_DIM = 512  # Fixed ArcFace embedding dimension (ResNet-100 output)

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id:     int = 0,
        det_size:   tuple[int, int] = (640, 640),
    ):
        self.model_name = model_name
        self.ctx_id     = ctx_id
        self._app       = None

        if not INSIGHTFACE_AVAILABLE:
            print("[ArcFaceExtractor] InsightFace unavailable. Returning zero embeddings.")
            return

        try:
            # allowed_modules: only load detection + recognition.
            # Excludes age/gender estimation — saves memory and load time.
            self._app = FaceAnalysis(
                name=model_name,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            # ctx_id=-1 forces CPU (no GPU on HuggingFace free tier)
            self._app.prepare(ctx_id=-1, det_size=det_size)
            print(f"[ArcFaceExtractor] Loaded model: {model_name}")
        except Exception as e:
            print(f"[ArcFaceExtractor] Failed to load model: {e}")
            self._app = None

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract 512-D ArcFace embedding from a face image.

        WHY INSIGHTFACE RUNS ITS OWN DETECTOR: Even though we pass an aligned
        112×112 crop, InsightFace's pipeline expects a full image and runs
        SCRFD (its internal detector) to locate the face region precisely.
        This double-detection step adds a small overhead but ensures the
        embedding model receives exactly the input format it was trained on.

        Parameters
        ----------
        face_img : np.ndarray
            Aligned BGR face image (112×112 from FaceAligner recommended).
            Any size is accepted — minimum 112×112 enforced internally.

        Returns
        -------
        np.ndarray of shape (512,), L2-normalized, dtype float32.
        Returns ZERO VECTOR if model unavailable or embedding fails — the
        fusion module handles zeros gracefully without crashing.
        """
        if self._app is None:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        img = self._ensure_bgr(face_img)          # Handle grayscale input
        img = self._ensure_min_size(img, 112)     # Upscale if smaller than 112px

        try:
            faces = self._app.get(img)  # Run InsightFace: detect + embed

            if faces:
                # If multiple faces detected, pick highest confidence (shouldn't happen
                # with an already-cropped 112×112 face, but guard against it)
                best = max(faces, key=lambda f: f.det_score)
                emb  = best.embedding
            else:
                # First attempt failed — pad image to 224×224 and retry
                # WHY: SCRFD detector sometimes misses faces that fill the entire frame
                # (no background context). Padding adds context that helps detection.
                img_padded = self._pad_to_square(img, target=224)
                faces = self._app.get(img_padded)
                if faces:
                    emb = max(faces, key=lambda f: f.det_score).embedding
                else:
                    # Both attempts failed — return zero vector
                    return np.zeros(self.FEATURE_DIM, dtype=np.float32)

            # Ensure float32 and L2-normalize to unit sphere
            # (InsightFace already normalizes, but re-normalizing is safe and explicit)
            emb  = emb.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb /= norm
            return emb

        except Exception as e:
            print(f"[ArcFaceExtractor] Extraction error: {e}")
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

    def extract_from_embedding_model(self, face_img: np.ndarray) -> np.ndarray:
        """
        Alternative: extract embedding using ONLY the recognition backbone,
        bypassing InsightFace's internal face detector.

        WHY THIS EXISTS: For already-tightly-cropped 112×112 faces (like our
        aligned crops), skipping the detector step can be slightly faster.
        However, this requires accessing InsightFace's internal model API which
        may change between versions. The main extract() method is more robust.

        Expects exactly 112×112 BGR input (ArcFace backbone requirement).
        """
        if self._app is None:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        img = cv2.resize(face_img, (112, 112))
        img = self._ensure_bgr(img)

        try:
            # Find the recognition model inside InsightFace's model collection
            rec_model = None
            for model in self._app.models.values():
                if hasattr(model, 'get_feat'):
                    rec_model = model
                    break

            if rec_model is None:
                # Fall back to the standard method if internal API not available
                return self.extract(face_img)

            # Manually preprocess: BGR→RGB, normalize to [-1,1], transpose to NCHW
            # (matches ArcFace training preprocessing from the original paper)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob    = (img_rgb.astype(np.float32) - 127.5) / 128.0  # [-1, 1] range
            blob    = blob.transpose(2, 0, 1)[np.newaxis, :]         # (1, 3, 112, 112)

            emb  = rec_model.get_feat(blob).ravel().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb /= norm
            return emb

        except Exception:
            # If internal API changed, fall back to standard method
            return self.extract(face_img)

    # ── Helper Methods ────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_bgr(img: np.ndarray) -> np.ndarray:
        """Convert grayscale to BGR if needed (InsightFace expects 3 channels)."""
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def _ensure_min_size(img: np.ndarray, min_size: int = 112) -> np.ndarray:
        """
        Upscale image if smaller than min_size in any dimension.
        WHY: InsightFace's SCRFD detector requires faces to be at least ~20px
        in the input image. At 112px, this works fine for our aligned crops.
        For very small inputs, cubic upscaling preserves sharpness better
        than bilinear (INTER_CUBIC vs INTER_LINEAR).
        """
        h, w = img.shape[:2]
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h = max(min_size, int(h * scale))
            new_w = max(min_size, int(w * scale))
            img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def _pad_to_square(img: np.ndarray, target: int = 224) -> np.ndarray:
        """
        Pad image to a square of at least 'target' pixels on each side.
        WHY CENTRE-PAD with zeros: Adds background context without distorting
        face proportions. Zero padding is neutral (black) and won't confuse
        InsightFace's internal detector the way border replication would.
        """
        h, w  = img.shape[:2]
        size  = max(h, w, target)
        padded = np.zeros((size, size, 3), dtype=img.dtype)  # Black canvas
        # Centre the face on the canvas
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        padded[y_off:y_off + h, x_off:x_off + w] = img
        return padded

    @property
    def dim(self) -> int:
        """Feature vector length (always 512 for ArcFace buffalo_l)."""
        return self.FEATURE_DIM
