"""
geometry_features.py
--------------------
WHAT : Facial geometry feature extractor using dlib's 68-point landmark model.
WHY  : HOG and LBP are texture/gradient descriptors — they change with lighting
       and blur. Facial GEOMETRY (landmark ratios) captures the structural
       relationships between facial parts that are:
         - ILLUMINATION-INVARIANT: ratios of distances are unaffected by brightness
         - SCALE-INVARIANT: normalized by face width/height
         - IDENTITY-SPECIFIC: inter-ocular distance, nose width, jaw width ratios
           differ between people and remain relatively stable across UAV angles
           (up to ~45° yaw / 30° pitch)

       FEATURES EXTRACTED:
         1. Eye Aspect Ratio (EAR)       — eye shape (wide-eyed vs narrow)
         2. Mouth Aspect Ratio (MAR)     — mouth openness
         3. Inter-eye distance           — relative eye spacing
         4. Face width-to-height ratio   — face shape (round vs elongated)
         5. Nose width ratio             — nose width relative to face width
         6. Jaw width ratio              — inner jaw vs outer jaw width
         7. Eyebrow height ratio         — brow position on the face
         8. Philtrum ratio               — upper lip to nose distance
         9. 68 raw landmark coordinates  — normalized (x,y) positions [0,1]
        10. 15 pairwise Euclidean distances between key landmark pairs

       WHY DLIB INSTEAD OF MEDIAPIPE: dlib's 68-point predictor is the
       established benchmark model with well-known failure modes. It is
       deterministic and does not require internet access for the model.
       MediaPipe 468-point mesh gives more detail but adds complexity.

       NOTE: Geometry features degrade at extreme poses (>45° yaw, >30° pitch).
       PCA + SVM handles this gracefully — the geometry components simply
       contribute less in high-pose scenarios.

Reference: King (2009). dlib-ml: A Machine Learning Toolkit. JMLR 10.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("[GeometryExtractor] WARNING: dlib not installed. "
          "Geometry features will return zero vectors. "
          "Install with: pip install dlib")


# ── dlib 68-point landmark index groups ──────────────────────────────────────
# Indices follow the standard iBUG 300-W face annotation scheme
_LEFT_EYE   = list(range(36, 42))  # 6 points around left eye
_RIGHT_EYE  = list(range(42, 48))  # 6 points around right eye
_NOSE       = list(range(27, 36))  # 9 nose bridge + tip points
_MOUTH      = list(range(48, 68))  # 20 outer + inner lip points
_JAW        = list(range(0, 17))   # 17 jaw contour points
_LEFT_BROW  = list(range(17, 22))  # 5 left eyebrow points
_RIGHT_BROW = list(range(22, 27))  # 5 right eyebrow points

# ── Landmark pairs for pairwise distance features ────────────────────────────
# Chosen to capture medically-recognised facial proportions (golden ratio,
# neoclassical facial canons) that are identity-discriminative
_DISTANCE_PAIRS = [
    (36, 45),  # eye-to-eye width (inter-ocular distance)
    (27, 33),  # nose bridge length
    (48, 54),  # mouth width (corner to corner)
    (0, 16),   # jaw width (outer contour)
    (19, 24),  # brow-to-brow width
    (36, 39),  # left eye width
    (42, 45),  # right eye width
    (33, 51),  # nose tip to upper lip (philtrum)
    (8, 27),   # chin to nose bridge (lower face height)
    (8, 33),   # chin to nose tip
    (36, 48),  # left eye corner to left mouth corner
    (45, 54),  # right eye corner to right mouth corner
    (27, 8),   # nose bridge to chin (full face height proxy)
    (17, 26),  # brow endpoints (brow spread)
    (21, 22),  # inner brow gap (glabella distance)
]


class GeometryExtractor:
    """
    Extracts facial geometry features from dlib 68-point landmarks.

    Landmarks are detected on the aligned 112×112 face crop and then
    normalized to [0, 1] range to be scale-invariant.

    Parameters
    ----------
    predictor_path    : str or Path — path to shape_predictor_68_face_landmarks.dat
    use_raw_landmarks : bool — include the 68 normalized (x,y) coordinate pairs
    use_distances     : bool — include the 15 pairwise inter-landmark distances
    """

    def __init__(
        self,
        predictor_path:    str = "models/shape_predictor_68_face_landmarks.dat",
        use_raw_landmarks: bool = True,
        use_distances:     bool = True,
    ):
        self.predictor_path    = Path(predictor_path)
        self.use_raw_landmarks = use_raw_landmarks
        self.use_distances     = use_distances

        self._detector  = None  # dlib HOG-based frontal face detector
        self._predictor = None  # dlib 68-point shape predictor

        if not DLIB_AVAILABLE:
            pass  # Warning already printed at import
        elif self.predictor_path.exists():
            # dlib's frontal detector is fast and sufficient for aligned crops
            self._detector  = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(str(self.predictor_path))
        else:
            print(
                f"[GeometryExtractor] WARNING: predictor not found at {predictor_path}. "
                "Geometry features will return zero vectors. "
                "Run setup.py to download the model."
            )

        # Total feature vector length depends on which sub-features are enabled
        ratio_dim    = 8                                       # 8 scalar ratio features
        landmark_dim = 68 * 2 if use_raw_landmarks else 0     # 136 if enabled
        distance_dim = len(_DISTANCE_PAIRS) if use_distances else 0  # 15 if enabled
        self.feature_dim = ratio_dim + landmark_dim + distance_dim

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract geometry features from an aligned 112×112 face image.

        Returns a zero vector if the predictor is unavailable or landmarks
        cannot be detected (graceful degradation — the fusion module handles
        zero vectors without crashing; PCA will down-weight them).

        Returns
        -------
        np.ndarray of shape (feature_dim,), dtype float32.
        """
        if self._predictor is None:
            # Return zeros so the pipeline can continue without geometry features
            return np.zeros(self.feature_dim, dtype=np.float32)

        gray      = self._preprocess(face_img)
        landmarks = self._detect_landmarks(gray)

        if landmarks is None:
            # Landmark detection failed (extreme pose, very blurry face, etc.)
            return np.zeros(self.feature_dim, dtype=np.float32)

        pts = landmarks  # Normalized (68, 2) landmark array in [0, 1]
        features = []

        # ── 1. Ratio-based features (scale and illumination invariant) ────────
        features.append(self._eye_aspect_ratio(pts, _LEFT_EYE))   # Left EAR
        features.append(self._eye_aspect_ratio(pts, _RIGHT_EYE))  # Right EAR
        features.append(self._mouth_aspect_ratio(pts))             # Mouth openness
        features.append(self._inter_eye_distance(pts))             # Eye spacing
        features.append(self._face_width_height_ratio(pts))        # Face shape
        features.append(self._nose_width_ratio(pts))               # Nose width
        features.append(self._jaw_width_ratio(pts))                # Jaw shape
        features.append(self._brow_height_ratio(pts))              # Brow position

        # ── 2. Raw landmark coordinates (normalized to [0,1]) ─────────────────
        # Gives fine-grained position info; PCA will learn which coords matter most
        if self.use_raw_landmarks:
            features.append(pts.ravel())  # 68 points × 2 coords = 136 values

        # ── 3. Pairwise Euclidean distances between key landmark pairs ─────────
        # Each distance captures a specific facial proportion (see _DISTANCE_PAIRS above)
        if self.use_distances:
            dists = [
                np.linalg.norm(pts[i] - pts[j])
                for i, j in _DISTANCE_PAIRS
            ]
            features.append(np.array(dists, dtype=np.float32))

        # np.atleast_1d ensures scalar ratios are treated as 1-element arrays
        return np.concatenate([
            np.atleast_1d(f) for f in features
        ]).astype(np.float32)

    def _detect_landmarks(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Run dlib face detector + 68-point shape predictor on a grayscale image.

        WHY FALLBACK TO FULL IMAGE RECT: dlib's frontal detector may fail on
        aligned crops because the head is already centred (the detector looks
        for profile context). Using the whole image as the face rectangle
        lets the predictor run even when the detector fires nothing.

        Returns
        -------
        np.ndarray of shape (68, 2) — landmark coordinates normalized to [0,1],
        or None if the predictor is unavailable.
        """
        if not DLIB_AVAILABLE or self._detector is None or self._predictor is None:
            return None

        h, w = gray.shape

        # Try dlib's HOG-based frontal face detector first
        rects = self._detector(gray, 1)  # upsample_num_times=1 for small faces
        if not rects:
            # Fallback: treat the entire 112×112 crop as the face region
            rects = [dlib.rectangle(0, 0, w, h)]

        rect  = rects[0]                    # Use the first (best) detection
        shape = self._predictor(gray, rect)  # Run 68-point predictor inside rect

        # Extract all 68 landmark (x, y) coordinates as a float array
        pts = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(68)],
            dtype=np.float32
        )

        # Normalize to [0, 1] range so distances are scale-invariant
        pts[:, 0] /= w
        pts[:, 1] /= h
        return pts

    # ── Ratio Feature Computations ────────────────────────────────────────────
    # Each ratio is computed from normalized landmarks so it is scale-invariant.
    # Adding 1e-6 to denominators prevents division by zero for extreme poses.

    @staticmethod
    def _eye_aspect_ratio(pts: np.ndarray, eye_indices: list) -> float:
        """
        Eye Aspect Ratio (EAR) = average vertical eye opening / horizontal width.
        EAR ≈ 0.2–0.35 for open eyes; drops near 0 for closed eyes.
        Discriminative because people differ in natural eye openness.
        """
        p = pts[eye_indices]
        # Two vertical measurements averaged for stability
        vertical   = (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / 2.0
        horizontal = np.linalg.norm(p[0] - p[3])
        return float(vertical / (horizontal + 1e-6))

    @staticmethod
    def _mouth_aspect_ratio(pts: np.ndarray) -> float:
        """
        Mouth Aspect Ratio (MAR) = average vertical opening / horizontal width.
        Captures mouth shape (thin-lipped vs full-lipped, open vs closed).
        Three vertical measurements averaged for robustness.
        """
        vertical = (
            np.linalg.norm(pts[51] - pts[59]) +
            np.linalg.norm(pts[52] - pts[58]) +
            np.linalg.norm(pts[53] - pts[57])
        ) / 3.0
        horizontal = np.linalg.norm(pts[48] - pts[54])
        return float(vertical / (horizontal + 1e-6))

    @staticmethod
    def _inter_eye_distance(pts: np.ndarray) -> float:
        """
        Distance between eye centroids, normalized by face width.
        Captures inter-ocular spacing — one of the most identity-stable features.
        """
        left_centre  = pts[_LEFT_EYE].mean(axis=0)
        right_centre = pts[_RIGHT_EYE].mean(axis=0)
        face_width   = np.linalg.norm(pts[0] - pts[16])
        return float(np.linalg.norm(left_centre - right_centre) / (face_width + 1e-6))

    @staticmethod
    def _face_width_height_ratio(pts: np.ndarray) -> float:
        """
        Face width (jaw tip to jaw tip) / face height (brow to chin).
        Captures overall face shape: round vs. elongated.
        """
        width  = np.linalg.norm(pts[0] - pts[16])   # Outer jaw width
        height = np.linalg.norm(pts[8] - pts[27])   # Chin to nose bridge
        return float(width / (height + 1e-6))

    @staticmethod
    def _nose_width_ratio(pts: np.ndarray) -> float:
        """Nose base width normalized by face width — captures nose breadth."""
        nose_width = np.linalg.norm(pts[31] - pts[35])
        face_width = np.linalg.norm(pts[0]  - pts[16])
        return float(nose_width / (face_width + 1e-6))

    @staticmethod
    def _jaw_width_ratio(pts: np.ndarray) -> float:
        """
        Inner jaw width (pts 4–12) / outer jaw width (pts 0–16).
        Captures jaw narrowing — wide vs. pointed chin shapes.
        """
        inner_jaw = np.linalg.norm(pts[4] - pts[12])
        outer_jaw = np.linalg.norm(pts[0] - pts[16])
        return float(inner_jaw / (outer_jaw + 1e-6))

    @staticmethod
    def _brow_height_ratio(pts: np.ndarray) -> float:
        """
        Average brow position relative to face height.
        Low brows (close to eyes) vs. high brows are identity-discriminative.
        """
        left_brow_h  = pts[_LEFT_BROW].mean(axis=0)[1]    # Mean vertical position
        right_brow_h = pts[_RIGHT_BROW].mean(axis=0)[1]
        avg_brow_h   = (left_brow_h + right_brow_h) / 2.0
        face_height  = np.linalg.norm(pts[8] - pts[27])
        # Distance from chin (pts[8]) to brow, relative to face height
        return float((pts[8, 1] - avg_brow_h) / (face_height + 1e-6))

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale uint8 and resize to 112×112 for dlib."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if gray.shape != (112, 112):
            gray = cv2.resize(gray, (112, 112))
        return gray  # Keep as uint8 — dlib expects uint8 grayscale

    @property
    def dim(self) -> int:
        """Feature vector length for this extractor configuration."""
        return self.feature_dim
