"""
detection.py
------------
WHAT : InsightFace SCRFD-based face detection wrapper.
WHY  : Replaces MTCNN (2016) with InsightFace SCRFD (2021) for three reasons:
         1. ACCURACY  — SCRFD achieves 93% mAP on WiderFace vs MTCNN's 85%.
                         Especially better on small, low-resolution faces (UAV use case).
         2. SPEED     — 5× faster than MTCNN at the same accuracy level.
         3. NO TENSORFLOW — MTCNN requires TensorFlow (~600 MB). SCRFD runs
                         on ONNX Runtime which is already installed for ArcFace.
                         This enables cloud deployment on memory-constrained hosts.

       LANDMARKS: SCRFD returns the same 5-point landmarks as MTCNN:
         left_eye, right_eye, nose, mouth_left, mouth_right
         → The alignment step (alignment.py) is completely UNCHANGED.

       MODEL: Uses 'buffalo_l' detection model (det_10g.onnx, ~16 MB).
         This is the SAME model pack used by ArcFace feature extraction,
         so no extra download is required if ArcFace is already loaded.

Pipeline position: RAW IMAGE → [FaceDetector] → aligned face → features

Reference: Guo et al. (2021). Sample and Computation Redistribution for
           Efficient Face Detection. ICLR 2022.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np


class FaceDetector:
    """
    Wraps InsightFace SCRFD for robust face detection in UAV imagery.

    WHY THIS CLASS EXISTS:
      Centralizes detection logic so every pipeline component gets a consistent
      interface regardless of the underlying detector backend. The public API
      (detect, detect_largest, crop_face, draw_detections) is identical to the
      previous MTCNN wrapper — no changes needed in alignment.py or pipeline.py.
    """

    def __init__(self, min_face_size: int = 20, confidence_threshold: float = 0.5):
        """
        Parameters
        ----------
        min_face_size : int
            Minimum face width/height in pixels.
            Set low (20px) because UAV faces at 20–30m altitude can be tiny.
        confidence_threshold : float
            Minimum SCRFD detection score to accept.
            0.5 is the standard threshold for SCRFD (vs 0.85 for MTCNN,
            because SCRFD scores are calibrated differently).
        """
        self.min_face_size         = min_face_size
        self.confidence_threshold  = confidence_threshold
        self._app                  = None
        self._load_detector()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _load_detector(self) -> None:
        """
        Load InsightFace FaceAnalysis in detection-only mode.

        WHY allowed_modules=['detection']:
          We only need the SCRFD detector here. The ArcFace recognition model
          is loaded separately in arcface_features.py. Loading only detection
          saves ~250 MB of RAM that the full buffalo_l pack would otherwise use.

        WHY buffalo_l:
          Contains det_10g.onnx (SCRFD 10GFlops) — the most accurate detector
          in the InsightFace model zoo for small face detection.
          If already downloaded for ArcFace, no re-download occurs.
        """
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name="buffalo_l",            # Highest accuracy (det_10g.onnx, 16MB detector)
                allowed_modules=["detection"],   # detection only — ArcFace loaded separately
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            print("[FaceDetector] SCRFD (buffalo_l) loaded successfully.")
        except Exception as e:
            print(f"[FaceDetector] SCRFD load failed: {e}")
            self._app = None

    # ── Internal conversion ────────────────────────────────────────────────────

    def _to_mtcnn_format(self, faces: list) -> list[dict]:
        """
        Convert InsightFace face objects to the same dict format that MTCNN used:
            {
                'box':       [x, y, w, h],
                'confidence': float,
                'keypoints': {
                    'left_eye':   (x, y),
                    'right_eye':  (x, y),
                    'nose':       (x, y),
                    'mouth_left': (x, y),
                    'mouth_right':(x, y),
                }
            }

        WHY SAME FORMAT: alignment.py, pipeline.py, and every router that calls
        detect() expects this dict structure. Converting here means zero changes
        are needed anywhere else in the codebase.
        """
        results = []
        for face in faces:
            # bbox is [x1, y1, x2, y2]
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1

            # Skip faces smaller than the minimum size
            if w < self.min_face_size or h < self.min_face_size:
                continue
            # Skip low-confidence detections
            if float(face.det_score) < self.confidence_threshold:
                continue

            # Convert 5-point keypoints array → named dict (same keys as MTCNN)
            keypoints = {}
            if face.kps is not None and len(face.kps) >= 5:
                kps = face.kps.astype(int)
                keypoints = {
                    "left_eye":    (int(kps[0][0]), int(kps[0][1])),
                    "right_eye":   (int(kps[1][0]), int(kps[1][1])),
                    "nose":        (int(kps[2][0]), int(kps[2][1])),
                    "mouth_left":  (int(kps[3][0]), int(kps[3][1])),
                    "mouth_right": (int(kps[4][0]), int(kps[4][1])),
                }

            results.append({
                "box":        [x1, y1, w, h],
                "confidence": round(float(face.det_score), 4),
                "keypoints":  keypoints,
            })

        return results

    # ── Public API (identical to original MTCNN wrapper) ───────────────────────

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect all faces in a BGR image (OpenCV format).

        WHY NO BGR→RGB CONVERSION:
          Unlike MTCNN (trained on RGB), InsightFace/SCRFD expects BGR images
          (same format as OpenCV). No colour-channel conversion is needed.

        Parameters
        ----------
        image : np.ndarray — BGR image (H × W × 3)

        Returns
        -------
        list of dicts in MTCNN-compatible format (see _to_mtcnn_format).
        """
        if image is None or image.size == 0:
            return []
        if self._app is None:
            return []

        try:
            faces = self._app.get(image)
            return self._to_mtcnn_format(faces)
        except Exception as e:
            print(f"[FaceDetector] detect() error: {e}")
            return []

    def detect_largest(self, image: np.ndarray) -> dict | None:
        """
        Detect and return ONLY the largest face (by bounding box area).

        WHY LARGEST: In single-subject UAV scenarios the subject's face is
        almost always the largest detected region. This avoids false positives
        from background people or non-face regions.

        Returns
        -------
        dict or None — detection dict of the largest face, or None if no face found.
        """
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda f: f["box"][2] * f["box"][3])

    def crop_face(self, image: np.ndarray, box: list, padding: float = 0.20) -> np.ndarray | None:
        """
        Crop the face region from the image with optional context padding.

        WHY PADDING: Tight crops cut off chin, forehead, and ear information
        that classifiers rely on. A 20% padding gives the feature extractors
        (HOG, ArcFace) enough context without adding too much background noise.
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
        """
        Draw bounding boxes and keypoints on a copy of the image.
        Used for debugging / visual inspection of detections.
        Does NOT modify the original image (works on a copy).
        """
        vis = image.copy()
        for d in detections:
            x, y, w, h = d["box"]
            conf = d.get("confidence", 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"{conf:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            kp = d.get("keypoints", {})
            for pt in kp.values():
                cv2.circle(vis, pt, 3, (0, 0, 255), -1)
        return vis
