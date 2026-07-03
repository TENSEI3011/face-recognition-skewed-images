"""
alignment.py
------------
WHAT : Geometric face alignment — maps a detected face into a canonical
       112×112 crop with fixed eye positions using an affine similarity transform.
WHY  : Feature extractors (especially ArcFace) are trained on faces aligned to
       a specific template. Misaligned faces (rotated or scaled differently)
       produce embeddings that are less consistent, harming recognition accuracy.
       Alignment corrects in-plane rotation and scale so every face crop looks
       the same to the feature extractors.

       Note: This corrects IN-PLANE rotation only (roll).
       Out-of-plane effects (yaw/pitch — the primary UAV challenge) are handled
       separately via augmentation.py and the geometry extractor.

Pipeline position: detection output → [FaceAligner] → 112×112 aligned crop → features

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from typing import Optional


# ── Target eye positions on the 112×112 output face ──────────────────────────
# These coordinates are the ArcFace standard alignment template.
# WHY THESE SPECIFIC VALUES: They are the mean eye positions across millions of
# faces used to train ArcFace (buffalo_l on MS1MV3 dataset). Using these exact
# coordinates ensures our aligned crops match the distribution ArcFace was
# trained on, maximizing embedding quality.
LEFT_EYE_TARGET  = (38.2946, 51.6963)
RIGHT_EYE_TARGET = (73.5318, 51.5014)

OUTPUT_SIZE = (112, 112)  # ArcFace/InsightFace standard input size


class FaceAligner:
    """
    Aligns a detected face to a standard 112×112 crop using an affine
    similarity transform derived from the eye landmark positions.

    A similarity transform preserves shape (no shear/distortion) — it only
    applies rotation, uniform scaling, and translation. This is the correct
    transform for face alignment because faces are rigid objects.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = OUTPUT_SIZE,
        left_eye_target: tuple[float, float] = LEFT_EYE_TARGET,
        right_eye_target: tuple[float, float] = RIGHT_EYE_TARGET,
    ):
        self.output_size = output_size
        # Store as float32 arrays for OpenCV compatibility
        self.left_eye_target  = np.array(left_eye_target,  dtype=np.float32)
        self.right_eye_target = np.array(right_eye_target, dtype=np.float32)

    def align(
        self,
        image: np.ndarray,
        left_eye: tuple[int, int],
        right_eye: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Align a face in the full image using detected eye positions.

        WHY 2-POINT TRANSFORM: Using just left_eye and right_eye (2 points)
        is sufficient to define a similarity transform (4 DOF: tx, ty, scale,
        rotation). Adding more points would overconstrain the problem and
        introduce distortion if landmarks are slightly inaccurate.

        Parameters
        ----------
        image     : Full BGR image (not just the face crop)
        left_eye  : (x, y) pixel coordinate of left eye centre (from MTCNN)
        right_eye : (x, y) pixel coordinate of right eye centre

        Returns
        -------
        Aligned BGR face of shape output_size, or None if transform fails.
        """
        src = np.array([left_eye, right_eye], dtype=np.float32)  # source eye positions
        dst = np.array([self.left_eye_target, self.right_eye_target], dtype=np.float32)  # target

        M = self._estimate_similarity_transform(src, dst)
        if M is None:
            return None

        # INTER_LINEAR: bilinear interpolation — good quality/speed tradeoff for resize/warp
        # BORDER_REPLICATE: fills border pixels with edge values (avoids black borders
        # which would confuse HOG/LBP at the face boundary)
        aligned = cv2.warpAffine(
            image, M, self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return aligned

    def align_from_detection(
        self,
        image: np.ndarray,
        detection: dict,
    ) -> Optional[np.ndarray]:
        """
        Convenience wrapper: align directly from an MTCNN detection dictionary.

        WHY FALLBACK: MTCNN occasionally returns detections without keypoints
        (very low-resolution or partially occluded faces). In that case we fall
        back to a simple crop + resize. It's worse than proper alignment but
        still usable — better than dropping the detection entirely.

        Parameters
        ----------
        detection : dict returned by FaceDetector.detect()
                    (must have 'keypoints' with 'left_eye' and 'right_eye')
        """
        kp = detection.get('keypoints', {})
        left_eye  = kp.get('left_eye')
        right_eye = kp.get('right_eye')

        if left_eye is None or right_eye is None:
            # Fallback: crop bounding box and resize without alignment
            # This happens rarely (very small or occluded faces)
            box = detection.get('box')
            if box:
                x, y, w, h = box
                crop = image[max(0, y):y + h, max(0, x):x + w]
                if crop.size > 0:
                    return cv2.resize(crop, self.output_size)
            return None

        return self.align(image, left_eye, right_eye)

    @staticmethod
    def _estimate_similarity_transform(
        src: np.ndarray,
        dst: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Compute the 2×3 affine matrix for a similarity transform from
        2-point correspondences. Uses the closed-form least-squares solution.

        WHY CLOSED-FORM instead of cv2.estimateAffinePartial2D:
          - Faster (no RANSAC needed — we trust MTCNN landmarks)
          - More numerically stable with exactly 2 points
          - Explicitly enforces similarity constraints (no shear)

        The similarity transform has the form:
          [cos(θ)·s   sin(θ)·s   tx]
          [-sin(θ)·s  cos(θ)·s   ty]
        where s = scale, θ = rotation angle.

        Returns
        -------
        np.ndarray of shape (2, 3) — affine transform matrix, or None if degenerate.
        """
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Center both point sets around their means
        src_c = src - src_mean
        dst_c = dst - dst_mean

        # Scale factor: ratio of how much the destination points span
        # compared to the source points (squared norm)
        src_var = (src_c ** 2).sum()
        if src_var < 1e-6:
            return None  # Degenerate: both source points are the same (shouldn't happen)

        scale = (src_c * dst_c).sum() / src_var

        # Rotation angle derived from cross / dot product of centered vectors
        angle = np.arctan2(
            src_c[0, 0] * dst_c[0, 1] - src_c[0, 1] * dst_c[0, 0],
            src_c[0, 0] * dst_c[0, 0] + src_c[0, 1] * dst_c[0, 1]
        )

        # Combined scale * rotation components
        cos_a = scale * np.cos(angle)
        sin_a = scale * np.sin(angle)

        # Build the 2×3 affine matrix (rotation + scale + translation)
        M = np.array([
            [cos_a,  sin_a, dst_mean[0] - cos_a * src_mean[0] - sin_a * src_mean[1]],
            [-sin_a, cos_a, dst_mean[1] + sin_a * src_mean[0] - cos_a * src_mean[1]],
        ], dtype=np.float32)

        return M

    def compute_pose_angle(
        self,
        left_eye: tuple[int, int],
        right_eye: tuple[int, int],
    ) -> float:
        """
        Estimate in-plane head roll (degrees) from eye positions.

        A roll of 0° means the eyes are perfectly horizontal.
        Positive = right eye higher, negative = left eye higher.
        Used for pose metadata logging in experiments, not in the main pipeline.
        """
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        return np.degrees(np.arctan2(dy, dx))
