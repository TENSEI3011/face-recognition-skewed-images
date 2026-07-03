"""
hog_features.py
---------------
WHAT : HOG (Histogram of Oriented Gradients) descriptor extractor for face images.
WHY  : HOG captures edge direction and magnitude patterns across the face.
       It is robust to illumination changes (uses gradient ratios, not absolute
       pixel values) and captures structural/shape information (eye brows, nose
       bridge, jaw line) that is identity-discriminative.

       WHY HOG IN ADDITION TO ARCFACE: ArcFace (deep embedding) is excellent
       at frontal faces but degrades at extreme UAV angles (>60° yaw).
       HOG provides a complementary classical descriptor that is more
       predictable and interpretable in those edge cases.

       HOG PARAMETERS (tuned for 112×112 face images):
         orientations=9  : 9 gradient direction bins (0°–180°, unsigned)
                           More bins = finer angular resolution but more noise.
                           9 is the empirically optimal value from Dalal & Triggs.
         pixels_per_cell=(8,8) : Each cell is 8×8 pixels.
                           For 112×112 images, this gives 14×14 = 196 cells.
                           Smaller cells = more spatial detail but more noise.
         cells_per_block=(2,2) : Each block normalizes over 2×2 = 4 cells.
                           Block normalization (L2-Hys) removes lighting effects
                           by normalizing gradient magnitudes within each block.

Reference: Dalal & Triggs (2005). Histograms of Oriented Gradients for Human
           Detection. CVPR 2005. (Original HOG paper)

Face Recognition on Skewed UAV Images
"""

import cv2
import numpy as np
from skimage.feature import hog


class HOGExtractor:
    """
    Extracts HOG descriptor from an aligned 112×112 face image.

    WHY SKIMAGE.HOG instead of cv2.HOGDescriptor:
      scikit-image's HOG implementation exposes all parameters cleanly,
      supports block normalization methods (L2-Hys is standard for faces),
      and returns a flat feature vector directly. OpenCV's HOG is optimised
      for pedestrian detection with fixed parameters.

    Parameters
    ----------
    orientations    : int   — number of gradient direction bins (9 is standard)
    pixels_per_cell : (h,w) — size of each cell in pixels
    cells_per_block : (h,w) — number of cells per normalization block
    block_norm      : str   — normalization method.
                              'L2-Hys': L2-normalize then clip (standard for faces)
    """

    def __init__(
        self,
        orientations:    int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm:      str = "L2-Hys",
    ):
        self.orientations    = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm      = block_norm

        # Pre-compute expected feature length for a 112×112 input image
        # Formula: (n_blocks_x × n_blocks_y × cells_per_block_x × cells_per_block_y × orientations)
        n_cells_x  = 112 // pixels_per_cell[1]       # = 14
        n_cells_y  = 112 // pixels_per_cell[0]       # = 14
        n_blocks_x = n_cells_x - cells_per_block[1] + 1  # = 13
        n_blocks_y = n_cells_y - cells_per_block[0] + 1  # = 13
        self.feature_dim = (
            n_blocks_x * n_blocks_y
            * cells_per_block[0] * cells_per_block[1]
            * orientations
        )  # = 13 × 13 × 2 × 2 × 9 = 6084 for default settings

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract HOG descriptor from a face image.

        Parameters
        ----------
        face_img : np.ndarray
            BGR or grayscale image. Converted to grayscale internally.
            Any size is accepted — resized to 112×112 before extraction.

        Returns
        -------
        np.ndarray of shape (feature_dim,), dtype float32.
        """
        gray       = self._preprocess(face_img)

        descriptor = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=True,  # Return flat 1-D array (not spatial grid)
        )
        return descriptor.astype(np.float32)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale and apply CLAHE before HOG extraction.

        WHY GRAYSCALE: HOG operates on intensity gradients — color channels
        would triple computation and add noise without improving accuracy
        (face identity is encoded in shape, not color).

        WHY CLAHE (Contrast Limited Adaptive Histogram Equalization):
        UAV images have uneven illumination (bright sky, shadowed face).
        CLAHE locally equalises contrast across small image tiles, making
        gradient magnitudes more consistent under varying lighting conditions.
        This improves HOG robustness to outdoor illumination changes.
          clipLimit=2.0 : limits noise amplification in smooth regions
          tileGridSize=(8,8) : 8×8 tile matches HOG cell size for consistency
        """
        # Convert BGR → grayscale if color image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize to the expected 112×112 input size
        if gray.shape != (112, 112):
            gray = cv2.resize(gray, (112, 112))

        # Apply CLAHE for illumination normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        return gray

    @property
    def dim(self) -> int:
        """Feature vector length for this extractor configuration."""
        return self.feature_dim
