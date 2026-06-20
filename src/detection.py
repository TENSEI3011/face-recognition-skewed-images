"""
detection.py
------------
MTCNN-based face detection module.
Returns detected face bounding boxes and 5 facial landmarks
(left eye, right eye, nose, left mouth corner, right mouth corner).

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from mtcnn import MTCNN


class FaceDetector:
    """
    Wraps MTCNN for robust face detection.
    Handles small/low-resolution faces typical in UAV imagery.
    """

    def __init__(self, min_face_size: int = 20, confidence_threshold: float = 0.85):
        """
        Parameters
        ----------
        min_face_size : int
            Minimum face size in pixels. Lower value catches smaller faces at altitude.
        confidence_threshold : float
            Minimum detection confidence to retain a face.
        """
        self.detector = MTCNN(min_face_size=min_face_size)
        self.confidence_threshold = confidence_threshold

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR image (OpenCV format).

        Returns
        -------
        list of dicts, each with:
            'box'        : [x, y, w, h]
            'confidence' : float
            'keypoints'  : dict with 'left_eye', 'right_eye', 'nose',
                           'mouth_left', 'mouth_right'
        """
        if image is None or image.size == 0:
            return []

        # MTCNN expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)

        return [
            d for d in detections
            if d.get('confidence', 0) >= self.confidence_threshold
        ]

    def detect_largest(self, image: np.ndarray) -> dict | None:
        """
        Detect and return only the largest face (by bounding box area).
        Useful for single-subject UAV frames.

        Returns
        -------
        dict or None
        """
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f['box'][2] * f['box'][3])

    def crop_face(self, image: np.ndarray, box: list, padding: float = 0.20) -> np.ndarray | None:
        """
        Crop the face region with optional padding (fraction of face size).

        Parameters
        ----------
        padding : float
            Fractional padding around the bounding box (default 20%).

        Returns
        -------
        Cropped BGR face image, or None if box is invalid.
        """
        x, y, w, h = box
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    @staticmethod
    def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and keypoints on a copy of the image."""
        vis = image.copy()
        for d in detections:
            x, y, w, h = d['box']
            conf = d.get('confidence', 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"{conf:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            kp = d.get('keypoints', {})
            for pt in kp.values():
                cv2.circle(vis, pt, 3, (0, 0, 255), -1)
        return vis

