"""
hog_features.py
---------------
Histogram of Oriented Gradients (HOG) feature extractor.
Operates on aligned 112×112 face images.

HOG captures gradient structure (edge directions and magnitudes)
which is moderately robust to illumination changes.

Reference:
  Dalal & Triggs (2005). Histograms of Oriented Gradients for Human Detection.
  CVPR 2005.

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from skimage.feature import hog


class HOGExtractor:
    """
    Extracts HOG descriptor from an aligned face image.

    Default parameters are tuned for 112×112 face images.

    Parameters
    ----------
    orientations  : int   — number of gradient orientation bins
    pixels_per_cell : (h, w) — cell size in pixels
    cells_per_block  : (h, w) — block normalization size in cells
    block_norm    : str   — block normalization method ('L2-Hys' recommended)
    """

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
    ):
        self.orientations    = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm      = block_norm

        # Pre-compute expected feature length for 112×112 image
        n_cells_x = 112 // pixels_per_cell[1]
        n_cells_y = 112 // pixels_per_cell[0]
        n_blocks_x = n_cells_x - cells_per_block[1] + 1
        n_blocks_y = n_cells_y - cells_per_block[0] + 1
        self.feature_dim = (
            n_blocks_x * n_blocks_y * cells_per_block[0]
            * cells_per_block[1] * orientations
        )

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract HOG descriptor from a face image.

        Parameters
        ----------
        face_img : np.ndarray
            BGR or grayscale image. Will be converted to grayscale internally.
            Expected size: 112×112 (resized if different).

        Returns
        -------
        np.ndarray of shape (feature_dim,), dtype float32.
        """
        gray = self._preprocess(face_img)

        descriptor = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=True,
        )
        return descriptor.astype(np.float32)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and resize to 112×112."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if gray.shape != (112, 112):
            gray = cv2.resize(gray, (112, 112))

        # CLAHE equalization to handle illumination variation
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray

    @property
    def dim(self) -> int:
        return self.feature_dim

