"""
blink_detector.py
-----------------
WHAT : Active liveness challenge — blink detection using dlib 68-point
       face landmarks and Eye Aspect Ratio (EAR).

WHY  : Passive texture analysis can be fooled by high-quality phone screens.
       An active challenge (requiring the user to BLINK on command) defeats
       99% of photo / screen / video replay attacks because:
         - A static photo cannot blink.
         - A video cannot blink at a random, unpredictable moment.
         - Only a live person in front of the camera can pass.

HOW  : Eye Aspect Ratio (EAR) by Soukupová & Čech (2016).
         EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)
       where p1…p6 are the six eye landmark points for one eye.
       When the eye is open: EAR ≈ 0.30–0.40
       When the eye is closed (blink): EAR < 0.25
       A blink is a BRIEF drop below the threshold followed by recovery.

DEPENDENCY: dlib (already in requirements.txt as dlib-bin)
            shape_predictor_68_face_landmarks.dat (in models/)

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from scipy.spatial import distance as dist


# ── Landmark indices for eyes (dlib 68-point model) ─────────────────────────
# Right eye: indices 36-41  (facial right → image left)
# Left eye:  indices 42-47  (facial left  → image right)
RIGHT_EYE_IDX = list(range(36, 42))
LEFT_EYE_IDX  = list(range(42, 48))


def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    Parameters
    ----------
    eye_points : np.ndarray, shape (6, 2)
        The six (x, y) landmark coordinates of the eye.

    Returns
    -------
    float — EAR value. Open eye ≈ 0.30; closed eye < 0.25.
    """
    # Vertical distances (top-bottom pairs)
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Horizontal distance (corner-to-corner)
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C + 1e-8)
    return float(ear)


# ── Per-session challenge state ───────────────────────────────────────────────

@dataclass
class BlinkChallengeState:
    """Tracks the blink challenge state for one webcam session."""
    session_id:       str
    started_at:       float  = field(default_factory=time.time)
    timeout_sec:      float  = 7.0
    required_blinks:  int    = 1
    blinks_detected:  int    = 0
    passed:           bool   = False
    failed:           bool   = False
    # EAR tracking
    ear_history:      list   = field(default_factory=list)
    below_threshold:  bool   = False   # True while eye is closed in current blink
    # Smoothing
    ear_ema:          float  = 0.35    # exponential moving average of EAR
    ema_alpha:        float  = 0.3
    # Config
    ear_threshold:    float  = 0.25    # EAR below this → eye closed
    consec_frames:    int    = 0       # consecutive frames below threshold
    consec_required:  int    = 2       # min frames below threshold to count as blink

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.timeout_sec - self.elapsed)

    @property
    def is_active(self) -> bool:
        return not self.passed and not self.failed and self.elapsed < self.timeout_sec

    def mark_timeout(self):
        if not self.passed:
            self.failed = True


# ── Blink Detector ────────────────────────────────────────────────────────────

class BlinkDetector:
    """
    Active liveness blink detector using dlib 68-point facial landmarks.

    Usage:
        detector = BlinkDetector(predictor_path=...)

        # Start a new session
        state = detector.new_challenge("session-abc")

        # For each incoming webcam frame:
        result = detector.process_frame(frame_bgr, state)
        # result = {
        #     "ear": 0.28,
        #     "blinks": 0,
        #     "passed": False,
        #     "failed": False,
        #     "time_remaining": 5.3,
        # }
    """

    def __init__(
        self,
        predictor_path: Optional[str] = None,
        ear_threshold:  float = 0.25,
        consec_frames:  int   = 2,
        timeout_sec:    float = 7.0,
        required_blinks: int  = 1,
    ):
        self.ear_threshold   = ear_threshold
        self.consec_frames   = consec_frames
        self.timeout_sec     = timeout_sec
        self.required_blinks = required_blinks

        # Resolve predictor path
        if predictor_path is None:
            root = Path(__file__).resolve().parents[1]
            predictor_path = str(root / "models" / "shape_predictor_68_face_landmarks.dat")

        self._predictor_path = predictor_path
        self._detector  = None   # dlib frontal face detector
        self._predictor = None   # dlib shape predictor
        self._load_error: Optional[str] = None
        self._init_dlib()

    def _init_dlib(self):
        """Lazy-load dlib and the shape predictor model."""
        try:
            import dlib as _dlib
            self._detector  = _dlib.get_frontal_face_detector()
            if not Path(self._predictor_path).exists():
                self._load_error = (
                    f"Shape predictor not found: {self._predictor_path}. "
                    "Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                )
                print(f"[BlinkDetector] WARNING: {self._load_error}")
                return
            self._predictor = _dlib.shape_predictor(self._predictor_path)
            print(f"[BlinkDetector] Loaded shape predictor: {self._predictor_path}")
        except ImportError:
            self._load_error = "dlib not installed. Run: pip install dlib-bin"
            print(f"[BlinkDetector] WARNING: {self._load_error}")
        except Exception as e:
            self._load_error = str(e)
            print(f"[BlinkDetector] WARNING: {e}")

    @property
    def is_available(self) -> bool:
        """True if dlib loaded successfully and blink detection is functional."""
        return self._detector is not None and self._predictor is not None

    def new_challenge(
        self,
        session_id: str,
        timeout_sec: Optional[float] = None,
        required_blinks: Optional[int] = None,
    ) -> BlinkChallengeState:
        """Create a fresh blink challenge state for a new webcam session."""
        return BlinkChallengeState(
            session_id      = session_id,
            timeout_sec     = timeout_sec or self.timeout_sec,
            required_blinks = required_blinks or self.required_blinks,
            ear_threshold   = self.ear_threshold,
            consec_required = self.consec_frames,
        )

    def _get_landmarks(self, gray: np.ndarray):
        """
        Run dlib face detection + landmark prediction on a grayscale frame.
        Returns list of (6, 2) eye arrays: [(right_eye, left_eye), ...]
        """
        import dlib as _dlib
        dets = self._detector(gray, 0)
        result = []
        for det in dets:
            shape = self._predictor(gray, det)
            pts = np.array([[shape.part(i).x, shape.part(i).y]
                            for i in range(68)], dtype=np.float32)
            right_eye = pts[RIGHT_EYE_IDX]
            left_eye  = pts[LEFT_EYE_IDX]
            result.append((right_eye, left_eye))
        return result

    def compute_ear_from_frame(self, frame_bgr: np.ndarray) -> Optional[float]:
        """
        Compute the average EAR for the most prominent face in a frame.

        Returns None if no face / landmarks detected, or dlib is unavailable.
        """
        if not self.is_available:
            return None
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
            gray = cv2.resize(gray, (320, 240))   # fast detection size
            eye_pairs = self._get_landmarks(gray)
            if not eye_pairs:
                return None
            # Use the first (largest / most confident) face
            right_eye, left_eye = eye_pairs[0]
            ear = (
                _eye_aspect_ratio(right_eye) + _eye_aspect_ratio(left_eye)
            ) / 2.0
            return float(ear)
        except Exception:
            return None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        state: BlinkChallengeState,
    ) -> dict:
        """
        Process a single webcam frame against an active blink challenge.

        Updates `state` in place and returns a result dict:
          ear           : float | None — current EAR (None if no face)
          blinks        : int          — total blinks detected so far
          passed        : bool         — challenge passed
          failed        : bool         — challenge timed out / failed
          time_remaining: float        — seconds left
          message       : str          — human-readable status
        """
        # Check timeout
        if not state.is_active:
            if state.elapsed >= state.timeout_sec and not state.passed:
                state.mark_timeout()
            return {
                "ear":            None,
                "blinks":         state.blinks_detected,
                "passed":         state.passed,
                "failed":         state.failed,
                "time_remaining": state.time_remaining,
                "message":        "PASSED" if state.passed else "TIMED OUT — try again",
            }

        ear = self.compute_ear_from_frame(frame_bgr)

        if ear is not None:
            # Exponential moving average for smoother signal
            state.ear_ema = state.ema_alpha * ear + (1 - state.ema_alpha) * state.ear_ema
            state.ear_history.append(round(ear, 4))

            if ear < state.ear_threshold:
                state.consec_frames += 1
            else:
                # Eye just opened — if it was closed long enough, count a blink
                if state.consec_frames >= state.consec_required:
                    state.blinks_detected += 1
                    print(f"[BlinkDetector] Blink #{state.blinks_detected} detected "
                          f"(session={state.session_id}, EAR={ear:.3f})")
                state.consec_frames = 0

            # Check if enough blinks
            if state.blinks_detected >= state.required_blinks:
                state.passed = True

        # Final timeout check
        if state.elapsed >= state.timeout_sec and not state.passed:
            state.mark_timeout()

        return {
            "ear":             round(ear, 4) if ear is not None else None,
            "blinks":          state.blinks_detected,
            "passed":          state.passed,
            "failed":          state.failed,
            "time_remaining":  round(state.time_remaining, 1),
            "message":         (
                "✅ Liveness verified!" if state.passed
                else "❌ Timed out — try again" if state.failed
                else f"👁 Please blink naturally ({state.time_remaining:.0f}s remaining)"
            ),
        }
