"""
augmentation.py
---------------
WHAT : UAV-specific image degradation pipeline.
WHY  : Real drone cameras capture faces under conditions very different from
       studio/ground-level photos: motion blur from propeller vibration, sensor
       noise from small CMOS chips, JPEG artifacts from wireless streaming,
       brightness shifts from different altitudes, and resolution loss from
       distance. This module simulates all those effects so we can:
         1. Augment gallery images during training → more robust model
         2. Degrade probe images in experiments → measure how accuracy drops
            with altitude/motion (see run_degradation.py)
         3. Test the full pipeline without needing a real drone

WHY ALBUMENTATIONS: It applies transforms in a randomized, composable way
       with GPU-optional acceleration and correct handling of image types.
       We use it instead of manual cv2 calls because it handles edge cases
       (e.g., clipping pixel values after noise addition) automatically.

Pipeline position: Used optionally in load_dataset() and all experiment scripts.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
import albumentations as A
from enum import Enum


class DegradationProfile(Enum):
    """
    Predefined UAV-realistic degradation severity levels.

    Each level corresponds to a real operational scenario:
      CLEAN    → Hovering at 5m, calm wind, good light (ideal conditions)
      MILD     → 5–10m altitude, slight vibration
      MODERATE → 10–20m altitude, moderate wind / vibration
      SEVERE   → 20–30m altitude, significant shake and distance blur
      EXTREME  → >30m, very small face, heavy blur and noise
      MOTION   → Drone or subject moving fast (directional motion blur)
      COMBINED → Worst-case: all effects simultaneously
    """
    CLEAN    = "clean"     # Baseline — no degradation applied
    MILD     = "mild"
    MODERATE = "moderate"
    SEVERE   = "severe"
    EXTREME  = "extreme"
    MOTION   = "motion"
    COMBINED = "combined"


class UAVAugmentor:
    """
    Applies UAV-realistic image degradations to face images.

    Can be used in three ways:
      1. As a training augmentor: make gallery images more varied
      2. As a probe simulator: degrade test images to mimic UAV conditions
      3. As a sweep tool: iterate over all profiles to measure accuracy vs. severity

    Parameters
    ----------
    profile : DegradationProfile or str
        Severity profile to apply. Accepts enum or string (e.g., 'moderate').
    seed : int
        Random seed for reproducibility across runs.
    """

    def __init__(
        self,
        profile: DegradationProfile = DegradationProfile.MODERATE,
        seed: int = 42,
    ):
        # Accept both enum and string for convenience
        if isinstance(profile, str):
            profile = DegradationProfile(profile)
        self.profile = profile
        self.seed = seed
        # Pre-build the Albumentations Compose pipeline at init time
        # (avoids re-building per image call — much faster in batch use)
        self._transform = self._build_transform(profile)

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the configured degradation profile to a single BGR image.

        Always returns the same size as the input — internal downsampling
        is followed by upsampling to simulate resolution loss without
        changing the array shape (keeps downstream processing consistent).

        Parameters
        ----------
        image : np.ndarray — BGR image, any size

        Returns
        -------
        np.ndarray — degraded BGR image, same shape as input.
        """
        # CLEAN is identity — skip processing entirely (fast path)
        if self.profile == DegradationProfile.CLEAN:
            return image.copy()

        result = self._transform(image=image)
        return result["image"]

    def augment_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply the same degradation to a list of images (e.g., all probe images)."""
        return [self.augment(img) for img in images]

    @staticmethod
    def altitude_downsample(image: np.ndarray, altitude_m: float) -> np.ndarray:
        """
        Simulate resolution loss at a specific UAV altitude.

        WHY DOWNSAMPLE + UPSAMPLE: At higher altitudes, a face occupies fewer
        pixels on the sensor. We simulate this by shrinking the image to the
        approximate number of pixels the face would occupy at that altitude,
        then scaling back up. The upsampled image is blurry/pixelated, mimicking
        what the recognition system actually receives from a compressed/streamed
        UAV feed.

        Empirical pixel-width model (12MP camera, 60° HFOV):
          altitude_m → approximate face width in pixels
          5m  → ~80px  (very close, good resolution)
          10m → ~40px
          20m → ~20px
          30m → ~13px  (far, very poor resolution)

        Parameters
        ----------
        image      : np.ndarray — BGR face image
        altitude_m : float — UAV altitude in metres

        Returns
        -------
        np.ndarray — same shape as input, but with resolution loss applied.
        """
        face_px_map = {5: 80, 10: 40, 15: 27, 20: 20, 25: 16, 30: 13}
        altitudes = sorted(face_px_map.keys())
        pixels    = [face_px_map[a] for a in altitudes]

        # Linear interpolation between known altitude-pixel pairs
        face_px = int(np.interp(altitude_m, altitudes, pixels))
        face_px = max(8, face_px)  # Hard floor: 8px minimum to avoid degenerate arrays

        h, w = image.shape[:2]
        # Shrink to simulate distance → upsample back to original size
        # INTER_AREA: best quality for downsampling (anti-aliased)
        # INTER_CUBIC: bicubic upsampling (smoother than NEAREST, worse than Lanczos
        #              but much faster — acceptable for simulation purposes)
        small    = cv2.resize(image, (face_px, face_px), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h),              interpolation=cv2.INTER_CUBIC)
        return restored

    @staticmethod
    def apply_motion_blur(
        image: np.ndarray,
        kernel_size: int = 15,
        angle_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Apply directional motion blur to simulate drone movement.

        WHY DIRECTIONAL: UAV motion blur has a specific direction determined
        by the drone's flight path. A horizontal blur simulates forward flight;
        diagonal blur simulates banking. This is more realistic than isotropic
        Gaussian blur (which simulates defocus, not motion).

        Parameters
        ----------
        kernel_size : int   — blur length in pixels (larger = faster movement)
        angle_deg   : float — direction of motion in degrees (0 = horizontal)

        Returns
        -------
        np.ndarray — motion-blurred image, same shape as input.
        """
        # Create a 1D horizontal kernel (all weight on one row)
        k = np.zeros((kernel_size, kernel_size))
        k[kernel_size // 2, :] = 1.0 / kernel_size

        # Rotate the kernel to the desired motion angle
        M = cv2.getRotationMatrix2D(
            (kernel_size // 2, kernel_size // 2), angle_deg, 1.0
        )
        k = cv2.warpAffine(k, M, (kernel_size, kernel_size))

        # Re-normalize after rotation (warpAffine may alter the sum slightly)
        k = k / (k.sum() + 1e-8)

        return cv2.filter2D(image, -1, k)

    @staticmethod
    def _build_transform(profile: DegradationProfile) -> A.Compose:
        """
        Build the Albumentations Compose pipeline for a given profile.

        WHY PROBABILISTIC TRANSFORMS: Real UAV conditions are not constant
        — vibration, lighting, and compression vary frame-to-frame. Using
        probabilities (p=0.5, etc.) ensures the augmentation is stochastic,
        which prevents the model from overfitting to a single fixed degradation.

        Each transform is chosen to simulate a specific physical effect:
          GaussianBlur  → optical defocus from altitude / rotor vibration
          MotionBlur    → camera shake during drone movement
          GaussNoise    → sensor noise (higher at small ISO sensors)
          BrightnessContrast → illumination change with altitude/sun angle
          ImageCompression   → JPEG artifacts from wireless video streaming
          Downscale     → resolution loss from altitude (coarser simulation
                          than altitude_downsample; good for training augmentation)
        """

        if profile == DegradationProfile.MILD:
            # Low-level conditions: slight softening, minimal noise
            return A.Compose([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),           # Very soft blur
                A.GaussNoise(std_range=(0.01, 0.03), p=0.4),        # Light sensor noise
                A.RandomBrightnessContrast(                          # Slight lighting shift
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.ImageCompression(quality_range=(75, 90), p=0.3),  # Mild compression
            ])

        elif profile == DegradationProfile.MODERATE:
            # Mid-range conditions: noticeable but not severe
            return A.Compose([
                A.GaussianBlur(blur_limit=(3, 7), p=0.7),
                A.MotionBlur(blur_limit=(3, 9), p=0.5),             # Start adding motion blur
                A.GaussNoise(std_range=(0.02, 0.06), p=0.6),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ImageCompression(quality_range=(55, 75), p=0.5),
            ])

        elif profile == DegradationProfile.SEVERE:
            # Heavy conditions: strong blur, high noise, significant compression
            return A.Compose([
                A.GaussianBlur(blur_limit=(5, 11), p=0.8),
                A.MotionBlur(blur_limit=(7, 15), p=0.7),
                A.GaussNoise(std_range=(0.05, 0.12), p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.ImageCompression(quality_range=(40, 60), p=0.7),
                A.Downscale(scale_range=(0.3, 0.5), p=0.6),        # Resolution loss added
            ])

        elif profile == DegradationProfile.EXTREME:
            # Worst realistic conditions: >30m, heavy weather, fast movement
            return A.Compose([
                A.GaussianBlur(blur_limit=(9, 15), p=0.9),
                A.MotionBlur(blur_limit=(13, 21), p=0.8),
                A.GaussNoise(std_range=(0.10, 0.20), p=0.9),       # Strong noise
                A.RandomBrightnessContrast(
                    brightness_limit=0.4, contrast_limit=0.4, p=0.7),
                A.ImageCompression(quality_range=(30, 50), p=0.9),  # Heavy compression
                A.Downscale(scale_range=(0.15, 0.35), p=0.8),      # Severe resolution loss
            ])

        elif profile == DegradationProfile.MOTION:
            # Primarily motion-based: directional blur dominates
            # WHY NO DOWNSCALE: Motion doesn't reduce resolution — only distance does
            return A.Compose([
                A.MotionBlur(blur_limit=(11, 21), p=0.9),           # Strong directional blur
                A.GaussNoise(std_range=(0.03, 0.08), p=0.6),       # Some sensor noise
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            ])

        elif profile == DegradationProfile.COMBINED:
            # Worst-case combination: simulates all failure modes simultaneously
            # WHY OneOf FOR BLUR: In reality you get either defocus OR motion blur
            # (rarely both equally severe), so we pick one randomly per sample
            return A.Compose([
                A.OneOf([
                    A.GaussianBlur(blur_limit=(5, 13), p=1.0),
                    A.MotionBlur(blur_limit=(7, 17), p=1.0),
                ], p=0.85),
                A.GaussNoise(std_range=(0.04, 0.10), p=0.75),
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, contrast_limit=0.25, p=0.6),
                A.ImageCompression(quality_range=(40, 65), p=0.8),
                A.Downscale(scale_range=(0.25, 0.5), p=0.7),
            ])

        else:
            # CLEAN profile: identity transform (no-op Compose pipeline)
            return A.Compose([])
