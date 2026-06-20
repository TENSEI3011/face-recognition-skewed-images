"""
augmentation.py
---------------
UAV-specific image degradation pipeline using Albumentations.
Simulates real-world conditions encountered by drone cameras:

  1. Altitude simulation  → downsample + upsample (resolution loss)
  2. Motion blur          → linear kernel (drone movement)
  3. Gaussian blur        → rotor vibration / defocus
  4. Gaussian noise       → sensor noise
  5. JPEG compression     → streaming/transmission artifact
  6. Brightness/Contrast  → lighting variation across altitude
  7. Combined UAV profile → realistic worst-case combination

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
import albumentations as A
from enum import Enum


class DegradationProfile(Enum):
    """Predefined UAV-realistic degradation profiles."""
    CLEAN       = "clean"         # No degradation (baseline)
    MILD        = "mild"          # 5–10m altitude equivalent
    MODERATE    = "moderate"      # 10–20m altitude equivalent
    SEVERE      = "severe"        # 20–30m altitude equivalent
    EXTREME     = "extreme"       # >30m altitude equivalent
    MOTION      = "motion"        # Moving target on drone
    COMBINED    = "combined"      # Blur + noise + compression


class UAVAugmentor:
    """
    Applies UAV-realistic image degradations to face images.
    Used for:
      - Training data augmentation
      - Probe simulation from gallery images
      - Degradation sweep experiments

    Parameters
    ----------
    profile : DegradationProfile or str
        Severity profile to apply.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        profile: DegradationProfile = DegradationProfile.MODERATE,
        seed: int = 42,
    ):
        if isinstance(profile, str):
            profile = DegradationProfile(profile)
        self.profile = profile
        self.seed = seed
        self._transform = self._build_transform(profile)

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply degradation to a BGR image.

        Parameters
        ----------
        image : np.ndarray — BGR image (any size)

        Returns
        -------
        np.ndarray — degraded BGR image, same size as input.
        """
        if self.profile == DegradationProfile.CLEAN:
            return image.copy()
        result = self._transform(image=image)
        return result["image"]

    def augment_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply augmentation to a list of images."""
        return [self.augment(img) for img in images]

    @staticmethod
    def altitude_downsample(image: np.ndarray, altitude_m: float) -> np.ndarray:
        """
        Simulate resolution loss at a given UAV altitude.
        Approximates a 12MP camera at 60° HFOV.

        altitude_m → face width in pixels (empirical):
          5m  → ~80px, 10m → ~40px, 20m → ~20px, 30m → ~13px
        """
        face_px_map = {5: 80, 10: 40, 15: 27, 20: 20, 25: 16, 30: 13}
        # Linear interpolation for intermediate altitudes
        altitudes = sorted(face_px_map.keys())
        pixels = [face_px_map[a] for a in altitudes]
        face_px = int(np.interp(altitude_m, altitudes, pixels))
        face_px = max(8, face_px)  # Floor at 8px

        h, w = image.shape[:2]
        # Downsample to face_px size, then upsample back to original
        small = cv2.resize(image, (face_px, face_px), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        return restored

    @staticmethod
    def apply_motion_blur(
        image: np.ndarray,
        kernel_size: int = 15,
        angle_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Apply directional motion blur to simulate drone movement.

        Parameters
        ----------
        kernel_size : int   — blur length in pixels
        angle_deg   : float — direction of motion (degrees)
        """
        k = np.zeros((kernel_size, kernel_size))
        k[kernel_size // 2, :] = 1.0 / kernel_size

        # Rotate kernel to desired angle
        M = cv2.getRotationMatrix2D(
            (kernel_size // 2, kernel_size // 2), angle_deg, 1.0
        )
        k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
        k = k / (k.sum() + 1e-8)

        return cv2.filter2D(image, -1, k)

    @staticmethod
    def _build_transform(profile: DegradationProfile) -> A.Compose:
        """Return Albumentations Compose pipeline for the given profile."""

        if profile == DegradationProfile.MILD:
            return A.Compose([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.ImageCompression(quality_range=(75, 90), p=0.3),
            ])

        elif profile == DegradationProfile.MODERATE:
            return A.Compose([
                A.GaussianBlur(blur_limit=(3, 7), p=0.7),
                A.MotionBlur(blur_limit=(3, 9), p=0.5),
                A.GaussNoise(std_range=(0.02, 0.06), p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ImageCompression(quality_range=(55, 75), p=0.5),
            ])

        elif profile == DegradationProfile.SEVERE:
            return A.Compose([
                A.GaussianBlur(blur_limit=(5, 11), p=0.8),
                A.MotionBlur(blur_limit=(7, 15), p=0.7),
                A.GaussNoise(std_range=(0.05, 0.12), p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.ImageCompression(quality_range=(40, 60), p=0.7),
                A.Downscale(scale_range=(0.3, 0.5), p=0.6),
            ])

        elif profile == DegradationProfile.EXTREME:
            return A.Compose([
                A.GaussianBlur(blur_limit=(9, 15), p=0.9),
                A.MotionBlur(blur_limit=(13, 21), p=0.8),
                A.GaussNoise(std_range=(0.10, 0.20), p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
                A.ImageCompression(quality_range=(30, 50), p=0.9),
                A.Downscale(scale_range=(0.15, 0.35), p=0.8),
            ])

        elif profile == DegradationProfile.MOTION:
            return A.Compose([
                A.MotionBlur(blur_limit=(11, 21), p=0.9),
                A.GaussNoise(std_range=(0.03, 0.08), p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            ])

        elif profile == DegradationProfile.COMBINED:
            return A.Compose([
                A.OneOf([
                    A.GaussianBlur(blur_limit=(5, 13), p=1.0),
                    A.MotionBlur(blur_limit=(7, 17), p=1.0),
                ], p=0.85),
                A.GaussNoise(std_range=(0.04, 0.10), p=0.75),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
                A.ImageCompression(quality_range=(40, 65), p=0.8),
                A.Downscale(scale_range=(0.25, 0.5), p=0.7),
            ])

        else:
            # Clean — identity transform
            return A.Compose([])

