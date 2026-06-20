"""
lbp_features.py
---------------
Local Binary Pattern (LBP) feature extractor.
Uses uniform LBP on a grid of face regions for spatial encoding.

LBP encodes local micro-texture by comparing each pixel to its circular
neighbours. Grid-based LBP (face split into cells) captures both local
texture and its spatial distribution across the face.

Reference:
  Ahonen, Hadid & Pietikäinen (2006). Face Description with Local Binary
  Patterns: Application to Face Recognition. IEEE TPAMI 28(12).

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class LBPExtractor:
    """
    Grid-based LBP feature extractor.
    The face is divided into a uniform grid of cells.
    LBP histogram is computed per cell and concatenated.

    Parameters
    ----------
    n_points   : int  — number of neighbour points on circular LBP ring
    radius     : int  — radius of circular LBP neighbourhood
    method     : str  — LBP method ('uniform' recommended)
    grid_x     : int  — number of horizontal cells in face grid
    grid_y     : int  — number of vertical cells in face grid
    """

    def __init__(
        self,
        n_points: int = 8,
        radius: int = 1,
        method: str = "uniform",
        grid_x: int = 8,
        grid_y: int = 8,
    ):
        self.n_points  = n_points
        self.radius    = radius
        self.method    = method
        self.grid_x    = grid_x
        self.grid_y    = grid_y

        # Uniform LBP has n_points + 2 bins
        n_bins = n_points + 2 if method == "uniform" else 2 ** n_points
        self.feature_dim = grid_x * grid_y * n_bins
        self._n_bins = n_bins

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract grid-based LBP descriptor from a face image.

        Parameters
        ----------
        face_img : np.ndarray
            BGR or grayscale face image. Will be resized to 112×112.

        Returns
        -------
        np.ndarray of shape (feature_dim,), dtype float32.
        """
        gray = self._preprocess(face_img)
        lbp_image = local_binary_pattern(
            gray,
            P=self.n_points,
            R=self.radius,
            method=self.method,
        )
        return self._compute_grid_histogram(lbp_image)

    def _compute_grid_histogram(self, lbp_image: np.ndarray) -> np.ndarray:
        """Divide LBP image into grid cells and compute per-cell histograms."""
        h, w = lbp_image.shape
        cell_h = h // self.grid_y
        cell_w = w // self.grid_x

        descriptors = []
        for row in range(self.grid_y):
            for col in range(self.grid_x):
                y1 = row * cell_h
                y2 = y1 + cell_h
                x1 = col * cell_w
                x2 = x1 + cell_w
                cell = lbp_image[y1:y2, x1:x2]
                hist, _ = np.histogram(
                    cell.ravel(),
                    bins=self._n_bins,
                    range=(0, self._n_bins),
                    density=True,   # L1 normalize per cell
                )
                descriptors.append(hist)

        return np.concatenate(descriptors).astype(np.float32)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and resize to 112×112."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if gray.shape != (112, 112):
            gray = cv2.resize(gray, (112, 112))

        return gray.astype(np.float64)

    @property
    def dim(self) -> int:
        return self.feature_dim

