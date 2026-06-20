"""
arcface_features.py
-------------------
ArcFace deep face embedding extractor using InsightFace (ONNX runtime).
Produces a 512-dimensional L2-normalized identity embedding.

The pretrained model generalizes well across pose and lighting but
degrades at extreme angles (>60° yaw) without fine-tuning.

Reference:
  Deng et al. (2019). ArcFace: Additive Angular Margin Loss for Deep
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
          "Run: pip install insightface onnxruntime")


class ArcFaceExtractor:
    """
    Extracts 512-D ArcFace embeddings using InsightFace's pretrained
    buffalo_l model (ResNet-100 backbone, trained on MS1MV3).

    Parameters
    ----------
    model_name : str
        InsightFace model pack name. 'buffalo_l' is the high-accuracy model.
    ctx_id     : int
        Device ID. -1 = CPU, 0+ = GPU (CUDA). Default: 0 (GPU if available).
    det_size   : tuple
        Detection input size. Larger = slower but better for small faces.
    """

    FEATURE_DIM = 512

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = 0,
        det_size: tuple[int, int] = (640, 640),
    ):
        self.model_name = model_name
        self.ctx_id = ctx_id
        self._app = None

        if not INSIGHTFACE_AVAILABLE:
            print("[ArcFaceExtractor] InsightFace unavailable. "
                  "Returning zero embeddings.")
            return

        try:
            self._app = FaceAnalysis(
                name=model_name,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)
            print(f"[ArcFaceExtractor] Loaded model: {model_name}")
        except Exception as e:
            print(f"[ArcFaceExtractor] Failed to load model: {e}")
            self._app = None

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract 512-D ArcFace embedding from a face image.

        Attempts InsightFace detection internally. If no face found,
        uses the whole image crop as the face region.

        Parameters
        ----------
        face_img : np.ndarray
            Aligned BGR face image (112×112 recommended).

        Returns
        -------
        np.ndarray of shape (512,), L2-normalized, dtype float32.
        Returns zero vector if model unavailable or embedding fails.
        """
        if self._app is None:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # InsightFace expects BGR, full-frame (it runs its own detection)
        img = self._ensure_bgr(face_img)
        img = self._ensure_min_size(img, min_size=112)

        try:
            faces = self._app.get(img)
            if faces:
                # Return embedding of highest-confidence detection
                best = max(faces, key=lambda f: f.det_score)
                emb = best.embedding
            else:
                # Fallback: pad to larger size and retry
                img_padded = self._pad_to_square(img, target=224)
                faces = self._app.get(img_padded)
                if faces:
                    emb = max(faces, key=lambda f: f.det_score).embedding
                else:
                    return np.zeros(self.FEATURE_DIM, dtype=np.float32)

            # L2 normalize
            emb = emb.astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb /= norm
            return emb

        except Exception as e:
            print(f"[ArcFaceExtractor] Extraction error: {e}")
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

    def extract_from_embedding_model(self, face_img: np.ndarray) -> np.ndarray:
        """
        Alternative: extract embedding using only the recognition backbone
        (no internal detection step). Useful when face is already tightly cropped.

        Expects 112×112 BGR input.
        """
        if self._app is None:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        img = cv2.resize(face_img, (112, 112))
        img = self._ensure_bgr(img)

        try:
            # Access recognition model directly
            rec_model = None
            for model in self._app.models.values():
                if hasattr(model, 'get_feat'):
                    rec_model = model
                    break

            if rec_model is None:
                return self.extract(face_img)

            # Preprocess for ArcFace backbone
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob = (img_rgb.astype(np.float32) - 127.5) / 128.0
            blob = blob.transpose(2, 0, 1)[np.newaxis, :]  # NCHW

            emb = rec_model.get_feat(blob).ravel().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb /= norm
            return emb

        except Exception:
            return self.extract(face_img)

    @staticmethod
    def _ensure_bgr(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    @staticmethod
    def _ensure_min_size(img: np.ndarray, min_size: int = 112) -> np.ndarray:
        h, w = img.shape[:2]
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h = max(min_size, int(h * scale))
            new_w = max(min_size, int(w * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def _pad_to_square(img: np.ndarray, target: int = 224) -> np.ndarray:
        h, w = img.shape[:2]
        size = max(h, w, target)
        padded = np.zeros((size, size, 3), dtype=img.dtype)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        padded[y_off:y_off+h, x_off:x_off+w] = img
        return padded

    @property
    def dim(self) -> int:
        return self.FEATURE_DIM

