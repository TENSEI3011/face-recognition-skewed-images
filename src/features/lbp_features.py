"""
lbp_features.py
---------------
WHAT : Grid-based Local Binary Pattern (LBP) feature extractor.
WHY  : LBP encodes local micro-texture — how pixels relate to their circular
       neighbours. It is:
         - Computationally fast (bitwise comparisons, no multiplications)
         - Illumination-robust: only relative intensities matter (dark face
           in bright scene behaves the same as well-lit face)
         - Complementary to HOG: HOG captures gradient direction (edges/shape);
           LBP captures texture within regions (skin pores, wrinkles, hair follicles)

       WHY GRID-BASED (spatial LBP): A single global LBP histogram discards
       spatial information. Dividing the face into a grid (8×8 = 64 cells) and
       computing per-cell histograms preserves WHERE in the face each texture
       appears — a raised eyebrow has a different spatial signature than a smiling mouth.

       WHY 'UNIFORM' LBP: Uniform patterns (≤2 transitions in the binary code)
       account for ~90% of real texture patterns. The 'uniform' method maps all
       non-uniform patterns to a single bin, reducing histogram size from 256
       to n_points+2 = 10 bins, making the descriptor less noisy.

       LBP PARAMETERS (tuned for 112×112 face images):
         n_points=8  : 8 neighbours on a circle of radius 1 — the standard
                       configuration for face texture description.
         radius=1    : 1-pixel radius = captures fine micro-texture.
                       Larger radius captures coarser structure (can be combined).
         grid_x=8, grid_y=8 : 8×8 grid = 64 cells. Matches HOG cell count
                               for consistent spatial resolution comparison.

Reference: Ahonen, Hadid & Pietikäinen (2006). Face Description with Local Binary
           Patterns: Application to Face Recognition. IEEE TPAMI 28(12).

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class LBPExtractor:
    """
    Grid-based LBP feature extractor for 112×112 face images.

    Divides the face into an n×m grid of cells and concatenates
    per-cell LBP histograms into a single descriptor vector.

    Parameters
    ----------
    n_points : int  — number of neighbour points on the circular LBP ring
    radius   : int  — radius of the circular neighbourhood in pixels
    method   : str  — LBP variant ('uniform' recommended for faces)
    grid_x   : int  — number of horizontal grid cells
    grid_y   : int  — number of vertical grid cells
    """

    def __init__(
        self,
        n_points: int = 8,
        radius:   int = 1,
        method:   str = "uniform",
        grid_x:   int = 8,
        grid_y:   int = 8,
    ):
        self.n_points = n_points
        self.radius   = radius
        self.method   = method
        self.grid_x   = grid_x
        self.grid_y   = grid_y

        # Uniform LBP histogram has n_points + 2 bins:
        #   n_points possible uniform codes + 1 bin for non-uniform codes + boundary
        # Non-uniform LBP would have 2^n_points = 256 bins (much noisier)
        n_bins           = n_points + 2 if method == "uniform" else 2 ** n_points
        self.feature_dim = grid_x * grid_y * n_bins  # Total descriptor length
        self._n_bins     = n_bins

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract grid-based LBP descriptor from a face image.

        Parameters
        ----------
        face_img : np.ndarray
            BGR or grayscale face image. Resized to 112×112 internally.

        Returns
        -------
        np.ndarray of shape (feature_dim,), dtype float32.
        """
        gray = self._preprocess(face_img)

        # Compute LBP codes for every pixel in the 112×112 grayscale image
        # Result is a same-size image where each pixel value = its LBP code
        lbp_image = local_binary_pattern(
            gray,
            P=self.n_points,
            R=self.radius,
            method=self.method,
        )

        # Divide the LBP image into grid cells and build per-cell histograms
        return self._compute_grid_histogram(lbp_image)

    def _compute_grid_histogram(self, lbp_image: np.ndarray) -> np.ndarray:
        """
        Split the LBP-coded image into a spatial grid and histogram each cell.

        WHY PER-CELL HISTOGRAMS: A face-level global histogram loses all
        spatial information. Histogramming per cell (14×14 pixels each for
        8×8 grid on 112×112 image) encodes WHERE each texture pattern occurs.

        WHY density=True: L1-normalizes each cell's histogram (sums to 1).
        This makes the descriptor invariant to cell area differences and
        allows comparison between faces at different scales.
        """
        h, w   = lbp_image.shape
        cell_h = h // self.grid_y  # Height of each cell in pixels
        cell_w = w // self.grid_x  # Width  of each cell in pixels

        descriptors = []
        for row in range(self.grid_y):
            for col in range(self.grid_x):
                # Extract this cell's region from the LBP image
                y1 = row * cell_h
                y2 = y1 + cell_h
                x1 = col * cell_w
                x2 = x1 + cell_w
                cell = lbp_image[y1:y2, x1:x2]

                # Build normalized histogram of LBP codes in this cell
                hist, _ = np.histogram(
                    cell.ravel(),
                    bins=self._n_bins,
                    range=(0, self._n_bins),
                    density=True,  # L1-normalize: area under histogram = 1
                )
                descriptors.append(hist)

        # Concatenate all 64 cell histograms into one flat descriptor
        return np.concatenate(descriptors).astype(np.float32)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale and resize to 112×112.

        LBP operates on intensity values, so color information is discarded.
        No CLAHE here (unlike HOG) — LBP is inherently illumination-robust
        because it uses relative pixel comparisons, not absolute values.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if gray.shape != (112, 112):
            gray = cv2.resize(gray, (112, 112))

        # float64 required by skimage's local_binary_pattern
        return gray.astype(np.float64)

    @property
    def dim(self) -> int:
        """Feature vector length for this extractor configuration."""
        return self.feature_dim
