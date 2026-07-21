"""
liveness.py
-----------
WHAT : Passive anti-spoofing / liveness detector for face recognition.

WHY  : Without liveness detection, the system can be fooled by:
         - Printed photos held in front of the camera
         - Phone/laptop screen showing someone's face   ← IMPROVED
         - Video replay attacks                         ← IMPROVED

HOW  : Six complementary passive texture signals are combined:
         1. LBP Uniformity      — printed paper has repetitive halftone ink patterns;
                                  real skin has irregular micro-texture with many
                                  non-uniform LBP codes.
         2. FFT Frequency       — screens produce periodic moiré / scan-line artifacts
                                  visible as sharp spectral peaks in the frequency domain.
         3. Gradient Coherence  — a flat photo has nearly uniform gradient magnitude
                                  across its surface; a real 3D face has structured depth
                                  gradients that vary across facial regions.
         4. Color Naturalness   — screens over-saturate colours; printed ink clips
                                  highlights; real skin sits in a natural HSV saturation
                                  band with moderate peak spread.
         5. Specular Highlight  — screens / glossy photos produce a hard, bright, uniform
                                  specular hotspot. Real skin has soft, diffuse speculars.
         6. Chroma Smoothness   — screen pixels are arranged in a perfectly regular grid,
                                  making chroma channels unnaturally smooth/low-noise.
                                  Real skin has organic chroma variation.

OFFLINE: Uses only OpenCV + NumPy — no model download required.
SPEED  : ~5ms per 112×112 face crop on CPU.

References:
  Boulkenafet et al. (2017). "Face Anti-Spoofing Using Texture Analysis." IEEE TIFS.
  Li et al. (2004). "Live Face Detection Based on the Analysis of Fourier Spectra." SPIE.
  Zhang et al. (2012). "A Face Antispoofing Database with Diverse Attacks." IEEE ICB.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np


# ── LBP implementation (no external dependency) ─────────────────────────────

def _lbp_image(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Compute a basic uniform LBP image.

    For each pixel p, compare with n_points neighbours on a circle of given radius.
    If neighbour >= p, bit = 1, else bit = 0.
    The LBP code is the decimal value of those bits.
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    angles = [2 * np.pi * i / n_points for i in range(n_points)]
    offsets = [(int(round(-radius * np.sin(a))),
                int(round(radius  * np.cos(a)))) for a in angles]

    # Restrict to valid interior pixels only
    pad = radius
    for i, (dy, dx) in enumerate(offsets):
        ny = np.arange(pad, h - pad) + dy
        nx = np.arange(pad, w - pad) + dx
        # Clamp to image bounds
        ny = np.clip(ny, 0, h - 1)
        nx = np.clip(nx, 0, w - 1)
        neighbor = gray[np.ix_(ny, nx)]
        center   = gray[pad:h - pad, pad:w - pad]
        bit = (neighbor >= center).astype(np.uint8)
        lbp[pad:h - pad, pad:w - pad] |= (bit << i)

    return lbp


def _lbp_uniformity_score(face_gray: np.ndarray) -> float:
    """
    Compute LBP uniformity ratio as a liveness indicator.

    Real skin:  many non-uniform LBP codes (value range spread) due to
                organic micro-texture variation.
    Printed/screen: few uniform codes repeated in halftone / pixel grid
                patterns (distribution is peaked at a few values).

    Returns score in [0, 1] where higher = more likely REAL.
    """
    lbp = _lbp_image(face_gray)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(np.float32)
    hist_norm = hist / (hist.sum() + 1e-8)

    # Shannon entropy: real faces have higher entropy (more diverse LBP codes)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
    max_entropy = np.log2(256)   # = 8.0

    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


# ── FFT frequency analysis ───────────────────────────────────────────────────

def _fft_liveness_score(face_gray: np.ndarray) -> float:
    """
    Detect screen/monitor artifacts using FFT spectral analysis.

    Screens produce periodic interference patterns (moiré, scan-lines, pixel grids)
    that appear as sharp isolated peaks in the frequency spectrum.
    Real skin has a smooth, diffuse frequency distribution without sharp peaks.

    Returns score in [0, 1] where higher = more likely REAL.
    """
    # Resize to fixed size for consistent frequency analysis
    face_resized = cv2.resize(face_gray, (64, 64))

    # Apply Hanning window to reduce spectral leakage at image edges
    h, w = face_resized.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = face_resized.astype(np.float32) * window

    # 2D FFT and shift DC to centre
    fft = np.fft.fft2(windowed)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift) + 1e-8
    log_mag = np.log(magnitude)

    # Normalise to [0, 1]
    log_mag -= log_mag.min()
    log_mag /= (log_mag.max() + 1e-8)

    # Peak-to-mean ratio: screens have sharp isolated peaks (high ratio)
    # Real faces have smooth spectrum (low ratio)
    mean_mag = log_mag.mean()
    max_mag  = log_mag.max()
    peak_ratio = float(max_mag / (mean_mag + 1e-8))

    # Empirically: real faces peak_ratio ~ 3-6; screens ~ 8-20
    # Map: low ratio → real (score ≈ 1.0), high ratio → spoof (score ≈ 0.0)
    score = float(np.clip(1.0 - (peak_ratio - 3.0) / 12.0, 0.0, 1.0))
    return score


# ── Gradient coherence analysis ──────────────────────────────────────────────

def _gradient_coherence_score(face_gray: np.ndarray) -> float:
    """
    Measure gradient magnitude variance as a 3D-structure cue.

    A flat printed photo / screen has nearly uniform gradient magnitude because
    it lacks 3D depth. A real face has strong structural gradients at nose
    bridge, eye sockets, jawline, and softer regions in between — producing
    high variance in the gradient magnitude map.

    Returns score in [0, 1] where higher = more likely REAL.
    """
    # Sobel gradients in both directions
    gx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Coefficient of variation (std/mean) — scale-invariant variance measure
    mean_mag = mag.mean() + 1e-8
    std_mag  = mag.std()
    cov = float(std_mag / mean_mag)

    # Real faces: cov ~ 0.8–1.5; photos/screens: cov ~ 0.3–0.7
    score = float(np.clip((cov - 0.3) / 1.0, 0.0, 1.0))
    return score


# ── Color naturalness analysis ────────────────────────────────────────────────

def _color_naturalness_score(face_bgr: np.ndarray) -> float:
    """
    Detect unnatural colour saturation from screens and glossy prints.

    Real skin occupies a specific HSV saturation band (S ≈ 40–160 for typical
    indoor lighting). Phone / laptop screens over-saturate colours — they push
    blues, reds, and greens to extremes to appear vivid. Printed photos clip
    highlights (very high V) while crushing saturation in shadow areas.

    We measure two things:
      - Mean saturation: screens tend to be too high (over-saturated)
      - Highlight clipping ratio: fraction of pixels with V > 240 (blown out
        specular / screen backlight bleed)

    Returns score in [0, 1] where higher = more likely REAL.
    """
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        return 0.5   # grayscale input — neutral

    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_channel = hsv[:, :, 1]   # 0–255
    v_channel = hsv[:, :, 2]   # 0–255

    mean_sat = float(s_channel.mean())

    # Real skin: mean saturation 40–140; screens: often 150–220
    # Clip below 40 → washed out (print); above 150 → over-saturated (screen)
    # Map the "natural" range to score ≈ 1.0
    if mean_sat < 20:
        sat_score = 0.30   # too desaturated → likely B&W print
    elif mean_sat < 40:
        sat_score = float(np.clip((mean_sat - 20) / 20, 0.0, 1.0)) * 0.7 + 0.3
    elif mean_sat <= 150:
        sat_score = 1.0    # natural skin range
    else:
        # Oversaturated — likely a screen; degrade score as saturation rises
        sat_score = float(np.clip(1.0 - (mean_sat - 150) / 100, 0.0, 1.0))

    # Highlight clipping: blown-out regions indicate screen backlight / reflection
    n_pixels = v_channel.size
    clipped   = float(np.sum(v_channel > 240)) / (n_pixels + 1e-8)
    clip_score = float(np.clip(1.0 - clipped * 15.0, 0.0, 1.0))

    return float(np.clip((sat_score + clip_score) / 2.0, 0.0, 1.0))


# ── Specular highlight analysis ───────────────────────────────────────────────

def _specular_highlight_score(face_gray: np.ndarray) -> float:
    """
    Detect hard, concentrated specular highlights that indicate a screen surface.

    A screen (phone / laptop / tablet) reflects ambient light as a single, very
    bright concentrated hotspot. Real skin (being a diffuse, micro-rough surface)
    produces soft, spread-out highlights.

    We detect the ratio of the area of extreme bright regions to total face area.
      - Real face: few extreme pixels, soft distribution
      - Screen:    compact bright region (the screen's glare point), or uniformly
                   bright surface with no dark regions (backlit phone)

    Returns score in [0, 1] where higher = more likely REAL.
    """
    # Threshold at top 2% brightness
    threshold = int(np.percentile(face_gray, 98))
    bright_mask = (face_gray >= threshold).astype(np.uint8)

    total_pixels  = face_gray.size
    bright_pixels = int(bright_mask.sum())
    bright_ratio  = bright_pixels / (total_pixels + 1e-8)

    # Find connected components of bright regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

    # If one dominant bright blob covers >5% of the face → hard specular (screen/glare)
    # Real skin: bright pixels are scattered (freckles, sweat, natural shine)
    max_blob_ratio = 0.0
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]   # skip background (label 0)
        max_blob_ratio = float(areas.max()) / (total_pixels + 1e-8)

    # Penalise concentrated bright blobs
    blob_score = float(np.clip(1.0 - max_blob_ratio * 20.0, 0.0, 1.0))

    # Also penalise if overall brightness distribution is too narrow (uniform backlit screen)
    bright_std = float(face_gray.std())
    # Real faces: std ~ 30–70; screens flat-lit: std ~ 5–20
    std_score = float(np.clip((bright_std - 10.0) / 40.0, 0.0, 1.0))

    return float(np.clip((blob_score + std_score) / 2.0, 0.0, 1.0))


# ── Chroma noise analysis ─────────────────────────────────────────────────────

def _chroma_noise_score(face_bgr: np.ndarray) -> float:
    """
    Detect the unnaturally smooth chroma of screens vs organic skin noise.

    Phone / laptop screens render pixels on a perfect rectangular grid. Even at
    webcam resolution, the colour channels of a screen face are slightly smoother
    than real skin (which has microscopic pore-level colour variation, melanin
    distribution, and subtle capillary blush patterns).

    We measure the standard deviation of a high-pass filtered chroma channel.
    Low chroma noise → screen/print (too smooth). High → real skin.

    Returns score in [0, 1] where higher = more likely REAL.
    """
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        return 0.5   # grayscale — neutral

    # Convert to YCrCb; Cr and Cb are chroma channels
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    # High-pass filter: subtract Gaussian blur to isolate fine chroma variation
    cr_blur = cv2.GaussianBlur(cr, (5, 5), 0)
    cb_blur = cv2.GaussianBlur(cb, (5, 5), 0)
    cr_hf = cr - cr_blur
    cb_hf = cb - cb_blur

    # Measure noise as std of high-frequency chroma
    chroma_noise = float((cr_hf.std() + cb_hf.std()) / 2.0)

    # Real faces: chroma_noise ~ 2.0–6.0; screens: ~ 0.3–1.5
    score = float(np.clip((chroma_noise - 0.5) / 4.0, 0.0, 1.0))
    return score


# ── Passive Liveness Detector ────────────────────────────────────────────────

class PassiveLivenessDetector:
    """
    Passive liveness detector using texture, frequency, gradient, colour, and
    specular analysis — covering both print and screen presentation attacks.

    No external model download required. Runs in ~5ms per face crop on CPU.

    Signal overview (6 complementary signals):
      1. LBP Uniformity   — catches printed halftone patterns
      2. FFT Frequency    — catches CRT/LCD periodic artifacts
      3. Gradient Coherence — catches flat-photo depth absence
      4. Color Naturalness  — catches screen over-saturation & print clipping
      5. Specular Highlight — catches screen glare hotspot & backlight
      6. Chroma Noise       — catches screen pixel-grid smoothness

    Parameters
    ----------
    threshold : float
        Minimum fused score to classify as REAL. Below this = SPOOF.
        Default 0.45 (slightly permissive — the active blink check provides
        the hard gate for webcam mode; passive is the fallback for uploads).
        Raise to 0.55 for stricter passive-only security.

    weights : tuple of 6 floats
        Relative weights for the six signals.
        (w_lbp, w_fft, w_grad, w_color, w_specular, w_chroma)
        Defaults emphasise colour and chroma checks (most discriminative for
        phone screens) while keeping classic texture checks active.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        weights: tuple = (0.20, 0.15, 0.15, 0.20, 0.15, 0.15),
    ):
        self.threshold = threshold
        self.weights   = weights

    def predict(self, face_img: np.ndarray) -> tuple[float, str]:
        """
        Predict liveness from a face image.

        Parameters
        ----------
        face_img : np.ndarray
            BGR or grayscale face crop (any size, will be resized internally).

        Returns
        -------
        (score, label)
            score : float in [0.0, 1.0] — higher = more likely REAL
            label : str — "REAL" or "SPOOF"
        """
        if face_img is None or face_img.size == 0:
            return 0.0, "SPOOF"

        # Keep a colour copy for colour-based signals
        face_bgr = face_img.copy() if face_img.ndim == 3 else None

        # Convert to grayscale for texture analyses
        if face_img.ndim == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()

        # Resize to standard 80×80 for consistent feature extraction
        gray = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_LINEAR)
        if face_bgr is not None:
            face_bgr = cv2.resize(face_bgr, (80, 80), interpolation=cv2.INTER_LINEAR)

        # ── Compute the six signals ────────────────────────────────────────
        try:
            s_lbp  = _lbp_uniformity_score(gray)
        except Exception:
            s_lbp  = 0.5

        try:
            s_fft  = _fft_liveness_score(gray)
        except Exception:
            s_fft  = 0.5

        try:
            s_grad = _gradient_coherence_score(gray)
        except Exception:
            s_grad = 0.5

        try:
            s_color = _color_naturalness_score(face_bgr) if face_bgr is not None else 0.5
        except Exception:
            s_color = 0.5

        try:
            s_spec = _specular_highlight_score(gray)
        except Exception:
            s_spec = 0.5

        try:
            s_chroma = _chroma_noise_score(face_bgr) if face_bgr is not None else 0.5
        except Exception:
            s_chroma = 0.5

        # ── Weighted fusion ────────────────────────────────────────────────
        w_lbp, w_fft, w_grad, w_color, w_spec, w_chroma = self.weights
        score = float(
            w_lbp    * s_lbp    +
            w_fft    * s_fft    +
            w_grad   * s_grad   +
            w_color  * s_color  +
            w_spec   * s_spec   +
            w_chroma * s_chroma
        )
        score = float(np.clip(score, 0.0, 1.0))

        label = "REAL" if score >= self.threshold else "SPOOF"
        return score, label

    def predict_detailed(self, face_img: np.ndarray) -> dict:
        """
        Extended predict that returns all individual signal scores for debugging.

        Returns
        -------
        dict with keys: score, label, s_lbp, s_fft, s_grad, s_color, s_spec, s_chroma
        """
        if face_img is None or face_img.size == 0:
            return {"score": 0.0, "label": "SPOOF",
                    "s_lbp": 0.0, "s_fft": 0.0, "s_grad": 0.0,
                    "s_color": 0.0, "s_spec": 0.0, "s_chroma": 0.0}

        face_bgr = face_img.copy() if face_img.ndim == 3 else None
        if face_img.ndim == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()

        gray = cv2.resize(gray, (80, 80), interpolation=cv2.INTER_LINEAR)
        if face_bgr is not None:
            face_bgr = cv2.resize(face_bgr, (80, 80), interpolation=cv2.INTER_LINEAR)

        def _safe(fn, *args):
            try:
                return fn(*args)
            except Exception:
                return 0.5

        s_lbp    = _safe(_lbp_uniformity_score, gray)
        s_fft    = _safe(_fft_liveness_score, gray)
        s_grad   = _safe(_gradient_coherence_score, gray)
        s_color  = _safe(_color_naturalness_score, face_bgr) if face_bgr is not None else 0.5
        s_spec   = _safe(_specular_highlight_score, gray)
        s_chroma = _safe(_chroma_noise_score, face_bgr) if face_bgr is not None else 0.5

        w_lbp, w_fft, w_grad, w_color, w_spec, w_chroma = self.weights
        score = float(np.clip(
            w_lbp * s_lbp + w_fft * s_fft + w_grad * s_grad +
            w_color * s_color + w_spec * s_spec + w_chroma * s_chroma,
            0.0, 1.0
        ))
        label = "REAL" if score >= self.threshold else "SPOOF"

        return {
            "score":    round(score, 4),
            "label":    label,
            "s_lbp":    round(s_lbp, 4),
            "s_fft":    round(s_fft, 4),
            "s_grad":   round(s_grad, 4),
            "s_color":  round(s_color, 4),
            "s_spec":   round(s_spec, 4),
            "s_chroma": round(s_chroma, 4),
        }

    def is_live(self, face_img: np.ndarray) -> bool:
        """Convenience method — returns True if the face is classified REAL."""
        score, label = self.predict(face_img)
        return label == "REAL"
