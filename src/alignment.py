"""
alignment.py
------------
Geometric face alignment using eye-corner landmarks.
Aligns detected faces to a canonical 112×112 frame with fixed eye positions.

This corrects for in-plane rotation and scale variation.
Note: Out-of-plane (yaw/pitch) correction requires 3DMM — handled in augmentation.py.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from typing import Optional


# Target eye positions on 112×112 aligned face (ArcFace standard)
LEFT_EYE_TARGET  = (38.2946, 51.6963)
RIGHT_EYE_TARGET = (73.5318, 51.5014)

OUTPUT_SIZE = (112, 112)


class FaceAligner:
    """
    Aligns a detected face to a standard 112×112 crop using
    an affine transformation derived from eye landmark positions.
    """

    def __init__(
        self,
        output_size: tuple[int, int] = OUTPUT_SIZE,
        left_eye_target: tuple[float, float] = LEFT_EYE_TARGET,
        right_eye_target: tuple[float, float] = RIGHT_EYE_TARGET,
    ):
        self.output_size = output_size
        self.left_eye_target = np.array(left_eye_target, dtype=np.float32)
        self.right_eye_target = np.array(right_eye_target, dtype=np.float32)

    def align(
        self,
        image: np.ndarray,
        left_eye: tuple[int, int],
        right_eye: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Align face using eye-corner coordinates detected by MTCNN.

        Parameters
        ----------
        image     : Full BGR image
        left_eye  : (x, y) pixel coordinate of left eye
        right_eye : (x, y) pixel coordinate of right eye

        Returns
        -------
        Aligned BGR face image of size output_size, or None on failure.
        """
        src = np.array([left_eye, right_eye], dtype=np.float32)
        dst = np.array([self.left_eye_target, self.right_eye_target], dtype=np.float32)

        # Estimate similarity transform (rotation + scale + translation, no shear)
        M = self._estimate_similarity_transform(src, dst)
        if M is None:
            return None

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
        Convenience wrapper: align directly from an MTCNN detection dict.

        Parameters
        ----------
        detection : dict returned by FaceDetector.detect()
        """
        kp = detection.get('keypoints', {})
        left_eye  = kp.get('left_eye')
        right_eye = kp.get('right_eye')

        if left_eye is None or right_eye is None:
            # Fallback: crop and resize without alignment
            box = detection.get('box')
            if box:
                x, y, w, h = box
                crop = image[max(0, y):y+h, max(0, x):x+w]
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
        Compute 2×3 affine matrix for similarity transform (rotation + uniform scale + translation).
        Uses least-squares closed-form solution.
        """
        # Closed-form similarity transform from 2-point correspondence
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        src_c = src - src_mean
        dst_c = dst - dst_mean

        # Scale
        src_var = (src_c ** 2).sum()
        if src_var < 1e-6:
            return None
        scale = (src_c * dst_c).sum() / src_var

        # Rotation angle
        angle = np.arctan2(
            src_c[0, 0] * dst_c[0, 1] - src_c[0, 1] * dst_c[0, 0],
            src_c[0, 0] * dst_c[0, 0] + src_c[0, 1] * dst_c[0, 1]
        )

        cos_a = scale * np.cos(angle)
        sin_a = scale * np.sin(angle)

        # Build 2x3 affine matrix
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
        Estimate in-plane rotation angle (degrees) from eye positions.
        Useful for logging pose metadata.
        """
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        return np.degrees(np.arctan2(dy, dx))

