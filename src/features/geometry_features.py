"""
geometry_features.py
--------------------
Facial geometry feature extractor using dlib 68-point landmarks.
Computes landmark-derived ratios that are illumination-invariant
and partially robust to minor pose variation.

Features extracted:
  - Eye aspect ratio (EAR) — left and right
  - Mouth aspect ratio (MAR)
  - Nose-to-face width ratio
  - Inter-eye distance (normalized)
  - Face width-to-height ratio
  - Jaw width ratio
  - Eyebrow position ratio
  - Philtrum ratio (upper lip to nose distance)
  - 68 raw landmark coordinates (normalized to [0,1])
  - 15 pairwise Euclidean distances between key landmarks

Note: Geometry features degrade at extreme pose (>45° yaw or >30° pitch).
The pipeline's fusion step handles this gracefully via feature normalization
and PCA; ablation studies will quantify this degradation.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
import dlib
from pathlib import Path
from typing import Optional


# dlib landmark indices (68-point model)
_LEFT_EYE   = list(range(36, 42))
_RIGHT_EYE  = list(range(42, 48))
_NOSE       = list(range(27, 36))
_MOUTH      = list(range(48, 68))
_JAW        = list(range(0, 17))
_LEFT_BROW  = list(range(17, 22))
_RIGHT_BROW = list(range(22, 27))

# Key landmark pairs for pairwise distances
_DISTANCE_PAIRS = [
    (36, 45),   # eye-to-eye width
    (27, 33),   # nose bridge length
    (48, 54),   # mouth width
    (0, 16),    # jaw width
    (19, 24),   # brow-to-brow width
    (36, 39),   # left eye width
    (42, 45),   # right eye width
    (33, 51),   # nose tip to upper lip
    (8, 27),    # chin to nose bridge
    (8, 33),    # chin to nose tip
    (36, 48),   # left eye to left mouth corner
    (45, 54),   # right eye to right mouth corner
    (27, 8),    # nose to chin
    (17, 26),   # brow ends
    (21, 22),   # inner brow gap
]


class GeometryExtractor:
    """
    Extracts facial geometry features from dlib 68 landmarks.

    Parameters
    ----------
    predictor_path : str or Path
        Path to dlib's shape_predictor_68_face_landmarks.dat file.
        Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    use_raw_landmarks : bool
        If True, include normalized raw landmark (x,y) coordinates.
    use_distances : bool
        If True, include pairwise inter-landmark Euclidean distances.
    """

    def __init__(
        self,
        predictor_path: str = "models/shape_predictor_68_face_landmarks.dat",
        use_raw_landmarks: bool = True,
        use_distances: bool = True,
    ):
        self.predictor_path = Path(predictor_path)
        self.use_raw_landmarks = use_raw_landmarks
        self.use_distances = use_distances

        self._detector  = dlib.get_frontal_face_detector()
        self._predictor = None

        if self.predictor_path.exists():
            self._predictor = dlib.shape_predictor(str(self.predictor_path))
        else:
            print(
                f"[GeometryExtractor] WARNING: predictor not found at {predictor_path}. "
                "Geometry features will return zero vectors. "
                "Download shape_predictor_68_face_landmarks.dat from dlib.net."
            )

        # Compute feature dimension
        ratio_dim = 8
        landmark_dim = 68 * 2 if use_raw_landmarks else 0
        distance_dim = len(_DISTANCE_PAIRS) if use_distances else 0
        self.feature_dim = ratio_dim + landmark_dim + distance_dim

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract geometry features from an aligned 112×112 face image.

        Returns
        -------
        np.ndarray of shape (feature_dim,), dtype float32.
        Returns zero vector if landmarks cannot be detected.
        """
        if self._predictor is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        gray = self._preprocess(face_img)
        landmarks = self._detect_landmarks(gray)

        if landmarks is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        pts = landmarks  # shape (68, 2), normalized to [0, 1]
        features = []

        # === Ratio Features ===
        features.append(self._eye_aspect_ratio(pts, _LEFT_EYE))   # left EAR
        features.append(self._eye_aspect_ratio(pts, _RIGHT_EYE))  # right EAR
        features.append(self._mouth_aspect_ratio(pts))              # MAR
        features.append(self._inter_eye_distance(pts))             # normalized
        features.append(self._face_width_height_ratio(pts))
        features.append(self._nose_width_ratio(pts))
        features.append(self._jaw_width_ratio(pts))
        features.append(self._brow_height_ratio(pts))

        # === Raw Landmark Coordinates ===
        if self.use_raw_landmarks:
            features.append(pts.ravel())  # 68 * 2 = 136 values, already [0,1]

        # === Pairwise Distances ===
        if self.use_distances:
            dists = [
                np.linalg.norm(pts[i] - pts[j])
                for i, j in _DISTANCE_PAIRS
            ]
            features.append(np.array(dists, dtype=np.float32))

        return np.concatenate([
            np.atleast_1d(f) for f in features
        ]).astype(np.float32)

    def _detect_landmarks(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Run dlib face detector + landmark predictor. Returns (68, 2) normalized array."""
        h, w = gray.shape

        # Try dlib's frontal detector first
        rects = self._detector(gray, 1)
        if not rects:
            # Fallback: use whole image as face region
            rects = [dlib.rectangle(0, 0, w, h)]

        rect = rects[0]
        shape = self._predictor(gray, rect)
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)

        # Normalize to [0, 1]
        pts[:, 0] /= w
        pts[:, 1] /= h
        return pts

    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray, eye_indices: list) -> float:
        """EAR = vertical eye openness / horizontal eye width."""
        p = pts[eye_indices]
        vertical = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / 2.0
        horizontal = np.linalg.norm(p[0] - p[3])
        return float(vertical / (horizontal + 1e-6))

    @staticmethod
    def _mouth_aspect_ratio(pts: np.ndarray) -> float:
        """MAR = vertical mouth opening / horizontal mouth width."""
        vertical = (
            np.linalg.norm(pts[51] - pts[59]) +
            np.linalg.norm(pts[52] - pts[58]) +
            np.linalg.norm(pts[53] - pts[57])
        ) / 3.0
        horizontal = np.linalg.norm(pts[48] - pts[54])
        return float(vertical / (horizontal + 1e-6))

    @staticmethod
    def _inter_eye_distance(pts: np.ndarray) -> float:
        """Distance between eye centroids, normalized by face width."""
        left_center  = pts[_LEFT_EYE].mean(axis=0)
        right_center = pts[_RIGHT_EYE].mean(axis=0)
        face_width   = np.linalg.norm(pts[0] - pts[16])
        return float(np.linalg.norm(left_center - right_center) / (face_width + 1e-6))

    @staticmethod
    def _face_width_height_ratio(pts: np.ndarray) -> float:
        """Face width (cheekbones) / face height (brow to chin)."""
        width  = np.linalg.norm(pts[0] - pts[16])
        height = np.linalg.norm(pts[8] - pts[27])
        return float(width / (height + 1e-6))

    @staticmethod
    def _nose_width_ratio(pts: np.ndarray) -> float:
        """Nose base width normalized by face width."""
        nose_width = np.linalg.norm(pts[31] - pts[35])
        face_width = np.linalg.norm(pts[0] - pts[16])
        return float(nose_width / (face_width + 1e-6))

    @staticmethod
    def _jaw_width_ratio(pts: np.ndarray) -> float:
        """Jaw width (pts 4 to 12) normalized by full jaw width."""
        inner_jaw = np.linalg.norm(pts[4] - pts[12])
        outer_jaw = np.linalg.norm(pts[0] - pts[16])
        return float(inner_jaw / (outer_jaw + 1e-6))

    @staticmethod
    def _brow_height_ratio(pts: np.ndarray) -> float:
        """Average brow height relative to face height."""
        left_brow_h  = pts[_LEFT_BROW].mean(axis=0)[1]
        right_brow_h = pts[_RIGHT_BROW].mean(axis=0)[1]
        avg_brow_h   = (left_brow_h + right_brow_h) / 2.0
        face_height  = np.linalg.norm(pts[8] - pts[27])
        return float((pts[8, 1] - avg_brow_h) / (face_height + 1e-6))

    @property
    def dim(self) -> int:
        return self.feature_dim

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.shape != (112, 112):
            gray = cv2.resize(gray, (112, 112))
        return gray

