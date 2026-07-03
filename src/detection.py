"""
detection.py
------------
WHAT : MTCNN-based face detection wrapper.
WHY  : MTCNN (Multi-Task Cascaded Convolutional Network) is used instead of
       simpler detectors (Haar cascades, HOG+SVM) because it simultaneously
       detects faces AND returns 5-point facial landmarks (eyes, nose, mouth
       corners). Those landmarks are essential for the geometric alignment step
       that follows. MTCNN also handles small, low-resolution, and partially
       occluded faces better than classical approaches — critical for UAV imagery.

Pipeline position: RAW IMAGE → [FaceDetector] → aligned face → features

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from mtcnn import MTCNN


class FaceDetector:
    """
    Wraps MTCNN for robust face detection in UAV imagery.

    WHY THIS CLASS EXISTS:
      Rather than calling MTCNN directly throughout the codebase, this class
      centralizes detection logic (confidence filtering, BGR→RGB conversion,
      face cropping) so every pipeline component gets consistent behavior.
    """

    def __init__(self, min_face_size: int = 20, confidence_threshold: float = 0.85):
        """
        Parameters
        ----------
        min_face_size : int
            Minimum face width/height in pixels that MTCNN will detect.
            Set low (20px) because UAV faces at 20–30m altitude can be tiny.
            Larger values speed up detection but miss distant faces.
        confidence_threshold : float
            MTCNN returns a probability per detection. We reject detections
            below 0.85 to filter out false positives (hands, logos, clothing)
            that MTCNN sometimes fires on in complex backgrounds.
        """
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold

        # MTCNN API changed in v1.0.0: min_face_size was removed from constructor.
        # We try both signatures for compatibility across installed versions.
        try:
            self.detector = MTCNN()
        except TypeError:
            self.detector = MTCNN(min_face_size=min_face_size)

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect all faces in a BGR image (OpenCV format).

        WHY BGR→RGB: MTCNN was trained on RGB images (PyTorch convention).
        OpenCV loads images as BGR by default, so we must convert before
        passing to MTCNN or the color channels will be swapped, hurting accuracy.

        Parameters
        ----------
        image : np.ndarray — BGR image (H × W × 3)

        Returns
        -------
        list of dicts, each with:
            'box'        : [x, y, w, h]  — bounding box
            'confidence' : float          — detection probability
            'keypoints'  : dict with keys 'left_eye', 'right_eye', 'nose',
                           'mouth_left', 'mouth_right' — (x, y) pixel coords
        """
        if image is None or image.size == 0:
            return []  # Guard against corrupt/empty frames from video capture

        # MTCNN expects RGB; OpenCV gives BGR — must convert
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)

        # Filter out low-confidence detections (false positives)
        return [
            d for d in detections
            if d.get('confidence', 0) >= self.confidence_threshold
        ]

    def detect_largest(self, image: np.ndarray) -> dict | None:
        """
        Detect and return ONLY the largest face (by bounding box area).

        WHY LARGEST: In single-subject UAV scenarios (surveillance of one person)
        the subject's face is almost always the largest detected region.
        Taking the largest box avoids false positives from background people
        or MTCNN firing on non-face regions in the periphery.

        Returns
        -------
        dict or None — detection dict of the largest face, or None if no face found.
        """
        faces = self.detect(image)
        if not faces:
            return None
        # Sort by area (w * h) and pick the maximum
        return max(faces, key=lambda f: f['box'][2] * f['box'][3])

    def crop_face(self, image: np.ndarray, box: list, padding: float = 0.20) -> np.ndarray | None:
        """
        Crop the face region from the image with optional context padding.

        WHY PADDING: Tight crops cut off chin, forehead, and ear information
        that classifiers rely on. A 20% padding gives the feature extractors
        (HOG, ArcFace) enough context without adding too much background noise.

        Parameters
        ----------
        image   : np.ndarray — full BGR frame
        box     : [x, y, w, h] — bounding box from detect()
        padding : float — fractional padding around the box (default 20%)

        Returns
        -------
        Cropped BGR face image, or None if box is invalid or out of bounds.
        """
        x, y, w, h = box
        # Convert fractional padding to pixel amounts
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        # Clamp coordinates to image boundaries to avoid index errors
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None  # Degenerate box (e.g., face at image edge)

        return image[y1:y2, x1:x2]

    @staticmethod
    def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw bounding boxes and keypoints on a copy of the image.
        Used for debugging / visual inspection of detections.
        Does NOT modify the original image (works on a copy).
        """
        vis = image.copy()
        for d in detections:
            x, y, w, h = d['box']
            conf = d.get('confidence', 0)
            # Green box with confidence label
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"{conf:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # Red dots for each landmark keypoint
            kp = d.get('keypoints', {})
            for pt in kp.values():
                cv2.circle(vis, pt, 3, (0, 0, 255), -1)
        return vis
