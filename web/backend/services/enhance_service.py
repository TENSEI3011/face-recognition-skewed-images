"""
enhance_service.py — Image Enhancement Pre-processing Service

Applies a chain of classical image restoration techniques to recover
recognizable facial detail from degraded (blurry/foggy/low-contrast) images
BEFORE the face detector sees them.

Enhancement chain (applied in order):
  1. CLAHE          — boosts local contrast (fixes fog/haze/dim images)
  2. Dehazing       — Dark Channel Prior fog removal (He et al. 2009)
  3. Unsharp Mask   — standard sharpening to recover edge detail from mild blur
  4. Wiener-like    — frequency-domain deconvolution for motion blur recovery

WHY HERE and not in detection.py:
  Enhancement is optional and expensive (~50-100ms). Keeping it as a separate
  service means it can be toggled per-request without touching the detector.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np


# ── 1. CLAHE contrast enhancement ─────────────────────────────────────────────

def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    WHY: Fog and haze reduce local contrast evenly across the image.
    Standard histogram equalization over-amplifies noise. CLAHE works on
    small local tiles (tile_size×tile_size) so bright areas don't wash out
    dark ones. clip_limit prevents noise amplification from over-contrast.

    Works in LAB color space so only luminance (L channel) is equalized —
    color balance is preserved.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ── 2. Dark Channel Prior dehazing ─────────────────────────────────────────────

def _get_dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute dark channel of the image (min intensity over local patch)."""
    min_channel = np.min(image, axis=2)  # Per-pixel minimum across RGB channels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)
    return dark


def _get_atmosphere(image: np.ndarray, dark: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    """Estimate global atmospheric light A from the brightest dark-channel pixels."""
    h, w = dark.shape
    n_pixels = h * w
    n_top = max(1, int(n_pixels * top_percent))
    flat_dark = dark.flatten()
    flat_img = image.reshape(-1, 3)
    idx = np.argsort(flat_dark)[-n_top:]
    A = np.max(flat_img[idx], axis=0)
    return A.astype(np.float64)


def dehaze(image: np.ndarray, omega: float = 0.85, t_min: float = 0.1) -> np.ndarray:
    """
    He et al. (2009) Dark Channel Prior single-image dehazing.

    WHY DARK CHANNEL: In haze-free outdoor images, at least one color channel
    in a local patch is very dark (near zero). Haze adds a uniform bright offset
    that erases this dark channel. Reversing this gives the haze-free estimate.

    Parameters
    ----------
    omega : float — haze removal strength (0.85 = strong but keeps some depth cue)
    t_min : float — floor on transmission map to avoid division by near-zero

    Returns
    -------
    Dehazed BGR image (uint8).
    """
    img_f = image.astype(np.float64) / 255.0
    dark = _get_dark_channel((img_f * 255).astype(np.uint8))
    A = _get_atmosphere((img_f * 255).astype(np.uint8), dark)
    A_norm = A / 255.0

    # Transmission map: t(x) = 1 - omega * dark_channel(I/A)
    normalized = img_f / (A_norm + 1e-6)
    dark_norm = _get_dark_channel((np.clip(normalized, 0, 1) * 255).astype(np.uint8))
    transmission = 1.0 - omega * (dark_norm.astype(np.float64) / 255.0)
    transmission = np.clip(transmission, t_min, 1.0)
    t = transmission[:, :, np.newaxis]

    # Recover scene radiance: J = (I - A) / t + A
    J = (img_f - A_norm) / t + A_norm
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)


# ── 3. Unsharp Mask sharpening ─────────────────────────────────────────────────

def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0,
                 strength: float = 1.5) -> np.ndarray:
    """
    Unsharp masking: sharpened = original + strength * (original - blurred).

    WHY UNSHARP MASK vs. plain Laplacian: Unsharp masking is noise-controlled
    (the Gaussian blur smooths noise before amplifying edges). Pure Laplacian
    sharpening amplifies noise equally with edges, making grainy images worse.

    Parameters
    ----------
    kernel_size : int   — Gaussian kernel size (must be odd)
    sigma       : float — Gaussian sigma (controls blur radius)
    strength    : float — how much to add back (1.5 = moderate, 3.0 = aggressive)
    """
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ── 4. Wiener-like frequency-domain deconvolution ─────────────────────────────

def wiener_deblur(image: np.ndarray, kernel_size: int = 5,
                  noise_power: float = 0.01) -> np.ndarray:
    """
    Simplified Wiener filter in the frequency domain for motion/defocus blur.

    WHY WIENER: Unsharp masking is good for mild softness but amplifies
    ringing artifacts on heavier blur. The Wiener filter suppresses noise
    amplification via the noise_power (SNR) term, giving cleaner recovery.

    Parameters
    ----------
    kernel_size  : int   — assumed PSF (point spread function) size
    noise_power  : float — regularization; higher = smoother but less sharp
                           (0.001 = aggressive deblur, 0.05 = gentle)
    """
    # Assume a uniform (box) PSF — reasonable for mild motion/defocus blur
    psf = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    channels = cv2.split(image)
    out_channels = []
    for ch in channels:
        ch_f = ch.astype(np.float32) / 255.0
        h, w = ch_f.shape

        # Pad PSF to image size for frequency-domain convolution
        psf_padded = np.zeros((h, w), dtype=np.float32)
        ph, pw = psf.shape
        psf_padded[:ph, :pw] = psf
        # Shift PSF so its center is at (0,0) in frequency domain
        psf_padded = np.roll(psf_padded, -(ph // 2), axis=0)
        psf_padded = np.roll(psf_padded, -(pw // 2), axis=1)

        H = np.fft.fft2(psf_padded)
        G = np.fft.fft2(ch_f)

        # Wiener filter: F̂ = H* G / (|H|² + noise_power)
        H_conj = np.conj(H)
        F_hat = (H_conj * G) / (np.abs(H) ** 2 + noise_power)
        recovered = np.abs(np.fft.ifft2(F_hat))
        recovered = np.clip(recovered, 0, 1)
        out_channels.append((recovered * 255).astype(np.uint8))

    return cv2.merge(out_channels)


# ── Public API ─────────────────────────────────────────────────────────────────

def enhance(image: np.ndarray, mode: str = "auto") -> np.ndarray:
    """
    Apply the full enhancement chain to a BGR image.

    Modes
    -----
    "auto"   — Run all stages: CLAHE → dehaze → unsharp mask  (safe default)
    "sharp"  — CLAHE + unsharp mask only  (for blurry but not hazy images)
    "dehaze" — CLAHE + dark channel dehaze only (for foggy/hazy images)
    "full"   — All stages including Wiener deconvolution (slowest, most aggressive)

    Parameters
    ----------
    image : np.ndarray — BGR image (uint8)
    mode  : str        — enhancement mode (see above)

    Returns
    -------
    Enhanced BGR image (uint8), same shape as input.
    """
    if image is None or image.size == 0:
        return image

    try:
        result = image.copy()

        # Stage 1: CLAHE — always applied (fast, always helps contrast)
        result = apply_clahe(result)

        if mode in ("auto", "dehaze", "full"):
            # Stage 2: Dehazing — remove fog/haze veil
            result = dehaze(result)

        if mode in ("auto", "sharp", "full"):
            # Stage 3: Unsharp mask — sharpen edges
            result = unsharp_mask(result, kernel_size=5, sigma=1.0, strength=1.2)

        if mode == "full":
            # Stage 4: Wiener deconvolution — heavier blur recovery
            result = wiener_deblur(result, kernel_size=5, noise_power=0.01)

        return result

    except Exception as e:
        print(f"[Enhance] Enhancement failed: {e} — returning original image.")
        return image
