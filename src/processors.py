import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import time
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy
import pandas
import pydensecrf.densecrf as dcrf
import skimage
import sklearn

from pydensecrf.utils import unary_from_softmax
from sklearn.base import ClassifierMixin, clone


logger = logging.getLogger("Processors")


# TODO:
#   *   Currently the "transform" method is mean to extract color features only.
#       As a "neighborhood" extractor class, it could be extended to extract other types of features as well,
#       such as texture features (e.g., local binary patterns) or edge features (e.g., Sobel filters) from
#       the neighborhood. Think about <skimage.filters.rank> functions.
class NeighborhoodExtractor:
    """
    Extracts pixel-wise color features from a 3D image, considering neighboring pixels.

    Parameters
    ----------
    n_neighbors : int
        Number of neighboring pixels to consider for feature extraction.
        If 0, only the pixel value is considered, else the `n_neighbors` pixel neighbors are included.

    include_center : bool, optional
        Whether to include the center pixel value in the output feature vector. Default is True.
        By central pixel, we refer to the pixel at the input coordinates from which the neighbors are extracted.

    center_loc : str, optional
        Position of the center pixel value in the output feature vector.
        Ignored if `include_center` is False. Possible values are:

        - "middle" (default): the center pixel value is placed in the middle of the feature vector
        - "beginning": the center pixel value is placed at the beginning of the feature vector

    Notes
    -----
    *   The total size of the neighborhood matrix is (2*radius + 1)^2 where radius is defined as
        (n_neighbors + 1) ** 0.5 - 1) // 2`.
    """

    _SUPPORTED_NEIGHBORS = {0, 8, 24}
    _SUPPORTED_CENTER_LOCATIONS = {"middle", "beginning"}

    def __init__(
            self,
            n_neighbors: int,
            include_center: bool = True,
            center_loc: str = "middle"
    ):
        # Validate input parameters
        if n_neighbors not in self._SUPPORTED_NEIGHBORS:
            raise ValueError(
                f"Invalid number of neighbors: {n_neighbors}. "
                f"Possible values are: {self._SUPPORTED_NEIGHBORS}"
            )
        if include_center and center_loc not in self._SUPPORTED_CENTER_LOCATIONS:
            raise ValueError(
                f"Invalid center pixel position ({center_loc}). "
                f"Expected one of: {self._SUPPORTED_CENTER_LOCATIONS}."
            )

        # Positional indices of the central pixel from which its neighbors are extracted
        n_channels = 3
        center_index_start = (n_neighbors // 2) * n_channels
        center_index_end = center_index_start + n_channels

        self.n_neighbors = n_neighbors
        self.include_center = include_center
        self.center_loc = center_loc
        self.center_pixel_slice = slice(center_index_start, center_index_end)

    def __repr__(self):
        return (
            "NeighborhoodExtractor("
                f"n_neighbors={self.n_neighbors}, "
                f"include_center={self.include_center}, "
                f"center_loc='{self.center_loc}'"
            ")"
        )

    def __str__(self):
        return self.__repr__()

    def to_json(self):
        """Return the current configuration as a JSON-serializable dict."""
        return {
            "n_neighbors": self.n_neighbors,
            "include_center": self.include_center,
            "center_loc": self.center_loc
        }

    @staticmethod
    def _validate_input(
            image: numpy.ndarray,
            yx_coordinates: Optional[numpy.ndarray] = None,
            mask: Optional[numpy.ndarray] = None
    ):
        if yx_coordinates is not None and mask is not None:
            raise ValueError("Both `yx_coordinates` and `mask` cannot be provided at the same time.")

        # ---------------------
        # VALIDATE IMAGE
        # ---------------------
        if not isinstance(image, numpy.ndarray):
            raise ValueError(f"Image must be a numpy array. Got: {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"Image must be a 3D array as (height, width, channels). Got: {image.shape}")
        if image.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels (e.g., RGB). Got: {image.shape[2]} channels")

        # ---------------------
        # VALIDATE MASK
        # ---------------------
        if mask is not None:
            if not isinstance(mask, numpy.ndarray):
                raise ValueError(f"Mask must be a numpy array. Got: {type(mask)}")
            if mask.ndim != 2:
                raise ValueError(f"Mask must be a 2D array. Got: {mask.shape}")
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Mask shape {mask.shape} does not match image height and width {image.shape[:2]}")
            if mask.dtype != bool:
                raise ValueError(f"Mask must be a boolean array. Got dtype: {mask.dtype}")

        # ---------------------
        # VALIDATE COORDINATES
        # ---------------------
        height, width = image.shape[:2]
        if yx_coordinates is not None:
            if not isinstance(yx_coordinates, numpy.ndarray):
                raise ValueError(f"Coordinates must be a numpy array. Got: {type(yx_coordinates)}")
            if yx_coordinates.ndim != 2 or yx_coordinates.shape[1] != 2:
                raise ValueError(f"Coordinates must be a 2D array as (n_pixels, 2). Got: {yx_coordinates.shape}")
            if not issubclass(yx_coordinates.dtype.type, numpy.integer):
                raise ValueError(f"Coordinates must be of integer type. Got dtype: {yx_coordinates.dtype}")
            if len(yx_coordinates) == 0 or len(yx_coordinates) > height * width:
                raise ValueError(
                    f"Number of coordinates must be between 1 and total image pixels ({height * width}). "
                    f"Got: {len(yx_coordinates)} coordinate pairs."
                )

            y_coords, x_coords = yx_coordinates[:, 0], yx_coordinates[:, 1]
            if (y_coords < 0).any() or (y_coords >= height).any():
                raise ValueError(
                    f"Y-coordinates must be in the range [0, {height - 1}]. "
                    f"Got: [{y_coords.min()}, {y_coords.max()}]"
                )
            if (x_coords < 0).any() or (x_coords >= width).any():
                raise ValueError(
                    f"X-coordinates must be in the range [0, {width - 1}]. "
                    f"Got: [{x_coords.min()}, {x_coords.max()}]"
                )

    def get_color_names(self, color_space: str) -> List[str]:
        """
        Create color feature names based on the specified color space and number of neighbors.

        Notes
        -----
        *   The center pixel is included as neighbor number 0 (e.g., "R0", "G0", "B0" for "RGB"
            color space) and the neighbors are numbered sequentially from 1 to `n_neighbors`.
            See examples 1 and 2 below.

        *   If `include_center` is False, the center pixel names (e.g., "R0", "G0", "B0" for "RGB"
            color space) will be omitted from the output list. See example 3 below.

        *   If `include_center` is True the center pixel names will be based on the `center_loc`
            parameter. For `center_loc="middle"`, see example 2 below and for `center_loc="beginning"`,
            see example 1 below.

        Examples
        --------
        1. Center pixel included at the beginning of the feature vector:
        ::
            extractor = NeighborhoodExtractor(n_neighbors=8, include_center=True, center_loc="beginning")
            extractor.get_color_names(color_space="RGB")

            # Output:
            [
                "R0", "G0", "B0",   <-- Center pixel values
                "R1", "G1", "B1",
                "R2", "G2", "B2",
                ...,
                "R8", "G8", "B8"
            ]

        2. Center pixel included in the middle of the feature vector:
        ::

            extractor = NeighborhoodExtractor(n_neighbors=8, include_center=True, center_loc="middle")
            extractor.get_color_names(color_space="RGB")
            # Output:
            [
                "R1", "G1", "B1",
                "R2", "G2", "B2",
                "R3", "G3", "B3",
                "R4", "G4", "B4",
                "R0", "G0", "B0",   <-- Center pixel values
                "R5", "G5", "B5",
                "R6", "G6", "B6",
                "R7", "G7", "B7",
                "R8", "G8", "B8"
            ]

        3. Center pixel not included:
        ::
            extractor = NeighborhoodExtractor(n_neighbors=8, include_center=False)
            extractor.get_color_names(color_space="RGB")
            # Output:
            [
                "R1", "G1", "B1",
                "R2", "G2", "B2",
                "R3", "G3", "B3",
                ...,
                "R8", "G8", "B8"
            ]
        """
        # Check input parameters
        if len(color_space) != 3:
            raise ValueError(f"Invalid color space `{color_space}`. Must be a three characters string.")
        elif not color_space.isalpha():
            raise ValueError(f"Invalid color space `{color_space}`. Must be an alphabetic string.")

        # Get the color space characters
        char1, char2, char3 = [char for char in color_space.upper()]

        # Create the output name list (omitting the center pixel)
        color_names = []
        for neighbor_num in range(1, self.n_neighbors + 1):
            neighbor_str = str(neighbor_num)
            color_names.extend([
                f"{char1}{neighbor_str}",
                f"{char2}{neighbor_str}",
                f"{char3}{neighbor_str}"
            ])

        # Include the center pixel names based on the `include_center` and `center_loc` parameters
        if self.include_center:
            center_color_names = [f"{char1}0", f"{char2}0", f"{char3}0"]
            if self.center_loc == "middle":
                # Insert the center pixel names in the middle of the list
                middle_index = len(color_names) // 2
                color_names[middle_index:middle_index] = center_color_names
            else:
                # Add the center pixel names at the beginning of the list
                color_names = center_color_names + color_names

        return color_names

    def get_neighboring_pixels(self, image: numpy.ndarray, yx_coordinates: numpy.ndarray) -> numpy.ndarray:
        """
        Get the image color values of the neighbors for multiple pixels simultaneously, with padding.

        Parameters
        ----------
        image : numpy.ndarray
            3D image array of shape (height, width, channels).

        yx_coordinates : numpy.ndarray | list
            Pixel coordinates as (row, col) to get the neighboring pixels.
            For example, [ (row_1, col_1), (row_2, col_2), ..., (row_N, col_M) ].

        Returns
        -------
        values : numpy.ndarray:
            Neighboring pixel values for all input pixel coordinates (`yx_coordinates`).
            Output shape is as: `(n_coordinates, image_channels * (n_neighbors + 1))`

            ``n_coordinates`` describes the total pixel coordinates in the input `yx_coordinates`
            and `channels * (n_neighbors + 1)` contains the color value of each image pixel and its
            `n_neighbors` neighbors. See notes.

        Notes
        -----
        *  Each "row" (dimension 0) in the returned array corresponds to a pixel in the input image and
            their "columns" (dimension 1) represent the pixel values of its neighbors, including the
            center pixel value itself.

        *  Output values for dimension 1 are ordered sequentially with the center pixel being in the
            middle of the array.
        """
        # Radius to define the neighborhood.
        #  8 neighbors --> radius=1
        # 24 neighbors --> radius=2
        radius = int( ((self.n_neighbors + 1) ** 0.5 - 1) // 2 )

        # Pad the matrix to handle edge cases for all pixels
        padded_image = numpy.pad(
            array=image,
            pad_width=((radius, radius), (radius, radius), (0, 0)),
            mode="symmetric",
            # constant_values=0
        )

        # Adjust coordinates for the padded matrix. Coordinate [row, col] becomes --> [row + radius, col + radius]
        n_coordinates = len(yx_coordinates)
        padded_coordinates = yx_coordinates + radius    # Shape: (n_coordinates, 2)

        # Generate relative offsets for the neighborhood. Example for n_neighbors=8 (radius=1):
        # [[-1, -1], [-1, 0], [-1, 1],
        #  [ 0, -1], [ 0, 0], [ 0, 1],
        #  [ 1, -1], [ 1, 0], [ 1, 1]]
        offsets = numpy.arange(-radius, radius + 1)     # Shape: (radius*2 + 1, )
        offset_grid = numpy.array(numpy.meshgrid(       # Shape: (n_neighbors + 1, 2)
            offsets, offsets,
            indexing='ij'
        )).reshape(2, -1).T

        # Add a new axis for broadcasting
        expanded_padded_coordinates = padded_coordinates[:, numpy.newaxis, :]   # Shape: (n_coordinates, 1, 2)
        offset_grid_expanded = offset_grid[numpy.newaxis, :, :]                 # Shape: (1, n_neighbors + 1, 2)

        # Add the expanded arrays to compute neighbor coordinates
        # The 1st dimension (n_coordinates and 1) will "expand" to n_coordinates (by broadcasting).
        # The 2nd dimension (1 and n_neighbors+1) will "expand" to n_neighbors+1 (by broadcasting).
        # The 3rd dimension (2 and 2) matches directly, so the addition happens element-wise for the broadcasted arrays.
        # Example calculation for pixel coordinate [row=2, col=2] with n_neighbors=8 (radius=1):
        # [2, 2] + [[-1, -1], [-1, 0], [-1, 1],
        #           [ 0, -1], [ 0, 0], [ 0, 1],
        #           [ 1, -1], [ 1, 0], [ 1, 1]]
        # Resulting in:
        # [[1, 1], [1, 2], [1, 3],
        #  [2, 1], [2, 2], [2, 3],
        #  [3, 1], [3, 2], [3, 3]]
        neighbor_coords = expanded_padded_coordinates + offset_grid_expanded  # Shape: (n_coordinates, n_neighbors + 1, 2)

        # Extract RGB values using advanced indexing
        # Shape: (n_coordinates, n_neighbors + 1, channels)
        neighborhood_values = padded_image[neighbor_coords[..., 0], neighbor_coords[..., 1]]

        # Get the neighbor values of shape: (`n_coordinates`, channels * (n_neighbors + 1))
        # Neighbors are ordered sequentially with the center pixel in the middle of the array
        return neighborhood_values.reshape(n_coordinates, -1)

    # TODO:
    #   *   Accept single channel image. Maybe just add one dimension if input image is a 2D array (height, width)
    #       and treat it as a single channel image with shape (height, width, 1).
    def transform(
            self,
            image: numpy.ndarray,
            yx_coordinates: Optional[Union[numpy.ndarray, list]] = None,
            mask: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """
        Get the pixel-wise color features from a 3D image, considering neighboring pixels.

        Parameters
        ----------
        image : numpy.ndarray
            3D array of shape (height, width, channels)

        yx_coordinates : list, optional
            Pixel coordinates as (row, col) to get the neighboring pixels. If not provided and `mask`
            is None, all neighboring pixels are computed. Input example:
            ::
                [ (row1, col1), (row2, col2), ..., (rowN, colM) ].

        mask : numpy.ndarray, optional
            2D array of shape (image_height, image_width) defining pixels of the image to extract features.
            Ignored if `yx_coordinates` is provided. If not provided and `yx_coordinates` is None, all image
            pixels are computed.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape `(n_samples, n_features)`

            *   ``n_samples`` refers to the number of pixel coordinates in the input ``yx_coordinates`` or
                defined by the ``mask`` (> 0). If both are None, it corresponds to the total number of pixels
                in the input image, i.e., `height * width`.

            *   ``n_features`` corresponds to the number of neighborhood features extracted for each pixel
                coordinate. It is defined as: `image_channels * (n_neighbors + 1)` if `include_center` is True,
                and `image_channels * n_neighbors` if `include_center` is False. See `center_loc` parameter for
                the position of the center pixel values in the output feature vector.
        """
        # ---------------------
        # INPUT VALIDATION
        # ---------------------
        if isinstance(yx_coordinates, list):
            yx_coordinates = numpy.asarray(yx_coordinates)
        self._validate_input(image=image, yx_coordinates=yx_coordinates, mask=mask)

        logger.debug(f"Processing feature space with {self.n_neighbors} neighbors")
        logger.debug(f"Input image shape: {image.shape}")
        if yx_coordinates is None:
            # Get the pixel coordinates from the mask array
            if mask is not None:
                yx_coordinates = numpy.argwhere(mask > 0)
                logger.debug(f"Using coordinates from mask: {len(yx_coordinates)} pixels")
            else:
                logger.debug(f"Using all image pixels: {image.shape[0] * image.shape[1]} pixels")
        else:
            logger.debug(f"Using provided coordinates: {len(yx_coordinates)} pixels")

        # ---------------------
        # FEATURE EXTRACTION
        # ---------------------
        # Feature space without neighboring pixels
        if self.n_neighbors == 0:
            if yx_coordinates is not None:
                features = image[yx_coordinates[:, 0], yx_coordinates[:, 1]]
            else:
                features = image.reshape(-1, 3)

        # Feature space with neighboring pixels
        else:
            # Generate pixel coordinates as (row, column) for the entire image
            if yx_coordinates is None:
                height, width, _ = image.shape
                rows, cols = numpy.meshgrid(numpy.arange(height), numpy.arange(width), indexing='ij')
                yx_coordinates = numpy.stack((rows, cols), axis=-1).reshape(-1, 2)

            # Neighboring pixel values
            # Shape: (n_coordinates, image_channels * (n_neighbors + 1))
            pixel_values = self.get_neighboring_pixels(image=image, yx_coordinates=yx_coordinates)

            if not self.include_center:
                # Neighboring pixel values. Omit the center pixel values from the color features
                # Shape: (n_coordinates, image_channels * n_neighbors)
                features = numpy.delete(pixel_values, self.center_pixel_slice, axis=1)
            else:
                if self.center_loc == "middle":
                    # Center pixel values are already in the middle of the feature vector
                    features = pixel_values
                else:
                    # Move the center pixel values to the beginning of the feature vector
                    center_pixel_values = pixel_values[:, self.center_pixel_slice]
                    neighbor_pixel_values = numpy.delete(pixel_values, self.center_pixel_slice, axis=1)
                    features = numpy.concatenate((center_pixel_values, neighbor_pixel_values), axis=1)
                    # features = numpy.append(center_pixel_values, neighbor_pixel_values, axis=1)

        return features

# ==============================================================================
# PROCESSING BLOCKS
# ==============================================================================
class PreprocessingBlock:
    """
    Independent preprocessing block pipeline. See the method ``transform()`` for the
    applied sequence of transformations. Example use::

        block = PreprocessingBlock()
        block.set_colorspace("HSV").set_gaussian_sigma(sigma=1.5)
        preprocessed_image = block.transform(rgb_image)

    Notes
    -----
    *   Available builder methods are: ``set_colorspace()`` and ``set_gaussian_sigma()``.

    *   By default, the block applies no color space conversion (i.e., `colorspace="RGB"`) and
        no gaussian smoothing (i.e., `gaussian_sigma=0.0`).

    *   This block is created and owned by `SegmentationProcessor` class and is accessible
        through `processor.preprocessing`.
    """
    def __init__(self):
        self.colorspace: str = "RGB"
        self.gaussian_sigma: float = 0.0
        logger.debug(f"Created {self}")

    # ------------------------------------------------------------------
    # Builder methods (fluent interface)
    # ------------------------------------------------------------------
    def set_colorspace(self, colorspace: str) -> "PreprocessingBlock":
        """
        Set the colorspace conversion.
        Must be one of the supported by `skimage.convert_colorspace()` function.
        Use ``"RGB"`` to disable colorspace conversion (default).
        """
        self.colorspace = colorspace.upper()
        logger.debug(f"Updated {self}")
        return self

    def set_gaussian_sigma(self, sigma: float) -> "PreprocessingBlock":
        """
        Set the gaussian smoothing standard deviation applied channel-wise.
        Use ``0.0`` to disable smoothing (default).
        """
        if sigma < 0.0:
            raise ValueError("Gaussian sigma must be >= 0.0")
        self.gaussian_sigma = sigma
        logger.debug(f"Updated {self}")
        return self

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------
    def transform(self, rgb_image: numpy.ndarray) -> numpy.ndarray:
        """
        Apply the configured preprocessing pipeline to an RGB image in the following
        steps sequence:

        * Cast to ``float64`` (normalized to `[-1.0, 1.0]` or `[0, 1.0]` depending on dtype of input)
        * Convert colorspace (only if ``colorspace`` is not "RGB").
        * Apply per-channel gaussian blur (only if ``gaussian_sigma`` > 0).

        Parameters
        ----------
        rgb_image : numpy.ndarray
            Input RGB image of shape `(height, width, 3)` and data type: `uint8` or `float`.

        Returns
        -------
        numpy.ndarray
            Preprocessed image of type `float64` and shape `(height, width, 3)`.
        """
        logger.debug(f"[Start] {self.__class__.__name__}.transform()")

        img_float = skimage.util.img_as_float64(image=rgb_image, force_copy=True)

        if self.colorspace != "RGB":
            img_float = skimage.color.convert_colorspace(
                arr=img_float,
                fromspace="RGB",
                tospace=self.colorspace
            )

        if self.gaussian_sigma > 0.0:
            img_float = skimage.filters.gaussian(
                image=img_float,
                sigma=self.gaussian_sigma,
                channel_axis=-1
            )

        logger.debug(f"[Finish] {self.__class__.__name__}.transform()")
        return img_float

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        """Return the current configuration as a JSON-serializable dict."""
        return {
            "colorspace": self.colorspace,
            "gaussian_sigma": self.gaussian_sigma,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
                f"colorspace='{self.colorspace}', "
                f"gaussian_sigma={self.gaussian_sigma}"
            ")"
        )

class PostprocessingBlock:
    """
    Independent postprocessing block: majority-vote smoothing + color visualization.

    Owns **all** color-related configuration: class names, per-class RGB colors,
    ignore index, background color, and the majority filter footprint.

    Configure state via fluent builder methods, then call :meth:`apply` on an
    assembled label mask::

        block = PostprocessingBlock(
            colors=[[0, 0, 255], [255, 255, 0]]
        )
        block.set_majority_filter(numpy.ones((5, 5), dtype=bool))
             .set_background((10, 10, 10))

        result = block.apply(label_mask, roi_mask)
        # result["labels"]       → 2-D int32 array
        # result["color_labels"] → 3-D uint8 RGB array

    This block is created and owned by :class:`SegmentationProcessor` and is accessible
    through ``processor.postprocessing``.

    Parameters
    ----------
    colors : List[Sequence[int]]
        Ordered list of RGB colors (0–255 range), one per class. The index of each color
        corresponds to the integer label assigned to that class in the predicted label mask.
    """
    def __init__(self, colors: List[Sequence[int]]):
        # Check input colors
        for color in colors:
            self._check_color(color)
            if tuple(color) == (0, 0, 0):
                raise ValueError("Class colors cannot be black (0, 0, 0) as it is reserved for background.")

        self.colors: List[tuple] = [tuple(c) for c in colors]
        self.ignore_index: int = len(colors)
        self._bg_color: Tuple[int, int, int] = (0, 0, 0)
        self._majority_footprint: Optional[numpy.ndarray] = None
        self.label_to_color: Dict[int, List[int]] = {i: list(color) for i, color in enumerate(colors)}
        logger.debug(f"Created {self}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_color(color: Sequence[int]):
        logger.debug("[Start] _check_color")
        if len(color) != 3:
            raise ValueError(f"Color must be a sequence of 3 integers (R, G, B). Got: {color}")
        for c in color:
            if not (0 <= c <= 255):
                raise ValueError(f"Color values must be in the range [0, 255]. Got: {color}")
        logger.debug("[Finish] _check_color")

    def _check_predicted_proba(self, predicted_proba: numpy.ndarray):
        logger.debug("[Start] _check_predicted_proba")
        if not isinstance(predicted_proba, numpy.ndarray):
            raise ValueError(f"Predicted probabilities must be a numpy array. Got: {type(predicted_proba)}")
        if not issubclass(predicted_proba.dtype.type, numpy.floating):
            raise ValueError(
                f"Predicted probabilities must be a floating-point array. Got: {predicted_proba.dtype}"
            )
        if predicted_proba.ndim != 3:
            raise ValueError(
                f"Predicted probabilities must be a 3D array of shape (H, W, num_classes). "
                f"Got: {predicted_proba.shape}"
            )
        if predicted_proba.shape[2] != len(self.colors):
            raise ValueError(
                f"The last dimension of predicted probabilities must match the number of classes "
                f"({len(self.colors)}). Got: {predicted_proba.shape[2]} classes."
            )
        if (predicted_proba < 0).any() or (predicted_proba > 1).any():
            raise ValueError(
                f"Predicted probabilities must be in the range [0, 1]. "
                f"Got: [{predicted_proba.min()}, {predicted_proba.max()}]"
            )
        if not numpy.allclose(predicted_proba.sum(axis=-1), 1.0):
            raise ValueError("Predicted probabilities must sum to 1 across the last dimension (classes).")
        logger.debug("[Finish] _check_predicted_proba")

    # ------------------------------------------------------------------
    # Builder methods (fluent interface)
    # ------------------------------------------------------------------
    def set_background_color(self, color: Tuple[int, int, int]) -> "PostprocessingBlock":
        """Set the RGB color used for background / masked-out pixels in the visualization."""
        self._check_color(color)
        if color in self.colors:
            raise ValueError(f"Input color {color} conflicts with class colors: {self.colors}")
        self._bg_color = tuple(color)
        logger.debug(f"Updated {self}")
        return self

    def set_majority_filter(self, footprint: Optional[numpy.ndarray]) -> "PostprocessingBlock":
        """
        Set the structuring element for majority-vote label smoothing.
        Use ``None`` to disable filtering (default).
        """
        if footprint is not None:
            if not isinstance(footprint, numpy.ndarray):
                raise ValueError(f"Footprint must be a numpy array. Got: {type(footprint)}")
            if footprint.ndim != 2:
                raise ValueError(f"Footprint must be a 2D array. Got: {footprint.shape}")
        self._majority_footprint = footprint
        logger.debug(f"Updated {self}")
        return self

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------
    def _apply_majority_filter(
            self,
            predicted_labels: numpy.ndarray,
            roi_mask: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """
        Majority filtering of the predicted label mask, applied only inside the valid region
        defined by `roi_mask`.

        Parameters
        ----------
        predicted_labels : numpy.ndarray
            2-D integer array of shape (H, W) containing the predicted class labels for each pixel.

        roi_mask : numpy.ndarray, optional
            Boolean 2-D mask of shape (H, W) where `True` indicates pixels inside the valid region
            for filtering. If `None` (default), the majority filter is applied to all pixels in
            `predicted_labels`.

        Returns
        -------
        numpy.ndarray
            2-D integer array of shape (H, W) with the majority filter applied to the valid region.
             Pixels outside the valid region (where `roi_mask` is `False`) are set to `ignore_index`.
        """
        logger.debug("[Start] _apply_majority_filter")

        # No mask means all pixels were segmented
        if roi_mask is None:
            # Create a fake mask representing all pixels for consistency later
            roi_mask = numpy.ones_like(predicted_labels, dtype=bool)

        # Detect if any label is negative or with a value greater than uint16 max
        # as the skimage majority filter requires uint8 or uint16 input.
        if (predicted_labels < 0).any() or (predicted_labels > numpy.iinfo(numpy.uint16).max).any():
            raise ValueError(
                f"Label mask values outside the valid range for majority filter (uint16). "
                f"Expected labels in [0, {numpy.iinfo(numpy.uint16).max}]. "
                f"Got: [{predicted_labels.min()}, {predicted_labels.max()}]"
            )

        # Data type for majority filter input
        if predicted_labels.max() <= numpy.iinfo(numpy.uint8).max:
            filer_dtype = numpy.uint8
        else:
            filer_dtype = numpy.uint16

        # Class count for debugging
        uq_labels, counts = numpy.unique(predicted_labels, return_counts=True)

        # Apply majority filter
        predicted_labels = skimage.filters.rank.majority(
            image=predicted_labels.astype(dtype=filer_dtype, copy=True),
            footprint=self._majority_footprint,
            mask=roi_mask
        )

        # Ensure pixels originally outside the valid mask are reset to "ignore_index",
        # as filtering near boundaries might change their values.
        predicted_labels[~roi_mask] = self.ignore_index

        # Log only labels that changed
        uq_labels_refined, counts_refined = numpy.unique(predicted_labels, return_counts=True)
        for label in uq_labels_refined:
            count_before = counts[uq_labels == label][0] if label in uq_labels else 0
            count_after = counts_refined[uq_labels_refined == label][0]
            diff_ratio = (count_after - count_before) / count_before if count_before > 0 else float('inf')
            if count_before != count_after:
                logger.debug(
                    f"Label {label} refined: before={count_before}, "
                    f"after={count_after} ({diff_ratio:.1%})"
                )

        logger.debug("[Finish] _apply_majority_filter")
        return predicted_labels

    def apply(
            self,
            image: numpy.ndarray,
            predicted_proba: numpy.ndarray,
            roi_mask: Optional[numpy.ndarray] = None
    ) -> Dict[str, numpy.ndarray]:
        """
        Apply postprocessing to the predicted class probabilities.

        Parameters
        ----------
        image : numpy.ndarray
            Original RGB image of shape `(height, width, 3)` used for creating overlay image.

        predicted_proba : numpy.ndarray
            3-D array of shape `(height, width, n_classes)` containing the predicted class
            probabilities for each pixel. Values must be in the range [0, 1] and sum to 1
            across the last dimension (classes).

        roi_mask : numpy.ndarray, optional
            Boolean 2-D mask `(height, width)`. When provided:

            * The majority filter is applied only inside the valid region.
            * ``ignore_index`` pixels receive ``_bg_color`` in the visualization.

        Returns
        -------
        Dict[str, numpy.ndarray]
            Postprocessing results of numpy arrays containing the following data:

            *   ``labels``: 2-D integer array of shape `(height, width)` with the predicted
                class labels for each pixel. Pixels with the value of ``ignore_index`` indicate
                masked-out areas (if `roi_mask` is provided).

            *   ``color_labels``: 3-D uint8 RGB array of shape `(height, width, 3)` where each pixel's
                color corresponds to its predicted class label. Pixels with the value of ``ignore_index``
                receive the background color defined by ``_bg_color``.

            *   ``overlay``: 3-D uint8 RGB array of shape `(height, width, 3)` representing an overlay
                visualization of the predicted class labels on top of the original image.
        """
        logger.debug(f"[Start] {self.__class__.__name__}.apply()")

        # Check predicted probabilities
        self._check_predicted_proba(predicted_proba)

        # Labels data type
        n_labels = predicted_proba.shape[2] + 1  # +1 for ignore_index
        if n_labels <= numpy.iinfo(numpy.uint8).max:
            labels_dtype = numpy.uint8
        elif n_labels <= numpy.iinfo(numpy.uint16).max:
            labels_dtype = numpy.uint16
        else:
            labels_dtype = numpy.uint32

        # Predicted integer labels
        predicted_labels = numpy.argmax(predicted_proba, axis=-1).astype(labels_dtype)
        if roi_mask is not None:
            predicted_labels[~roi_mask] = self.ignore_index

        # Majority filter (optional)
        if self._majority_footprint is not None:
            predicted_labels = self._apply_majority_filter(predicted_labels, roi_mask)

        # Color visualization
        all_colors = self.colors + [self._bg_color]
        color_palette = numpy.array(all_colors, dtype=numpy.uint8)
        color_labels = color_palette[predicted_labels]

        # Overlay image
        colors_alpha = 0.5
        img_float = skimage.util.img_as_float(image, force_copy=True)
        color_labels_float = skimage.util.img_as_float(color_labels, force_copy=True)
        img_overlay = (img_float * (1.0 - colors_alpha)) + (color_labels_float * colors_alpha)
        img_overlay = skimage.util.img_as_ubyte(img_overlay)

        logger.debug(f"[Finish] {self.__class__.__name__}.apply()")
        return {
            "labels": predicted_labels,
            "color_labels": color_labels,
            "overlay": img_overlay
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        """Return the current configuration as a JSON-serializable dict."""
        return {
            "colors": [list(c) for c in self.colors],
            "ignore_index": self.ignore_index,
            "bg_color": list(self._bg_color),
            "majority_footprint": (
                self._majority_footprint.tolist()
                if self._majority_footprint is not None
                else None
            ),
        }

    def __repr__(self) -> str:
        fp_shape = (
            str(self._majority_footprint.shape)
            if self._majority_footprint is not None
            else None
        )
        return (
            f"{self.__class__.__name__}("
                f"colors={self.colors}, "
                f"ignore_index={self.ignore_index}, "
                f"bg_color={self._bg_color}, "
                f"majority_footprint_shape={fp_shape}"
            ")"
        )

# ==============================================================================
# SEGMENTATION PROCESSOR
# ==============================================================================
class SegmentationProcessor:
    """
    Pixel-wise image segmentation processor.

    Composed of two configurable internal blocks and an internal Random Forest
    classifier:

    * ``preprocessing``: `PreprocessingBlock` class (colorspace, gaussian blur)
    * ``postprocessing``: `PostprocessingBlock` class (majority filter, visualization)
    * ``classifier``: `RandomForestClassifier`` sklearn class (owned directly by the processor)

    Available public methods:

    * ``evaluate_classifier()``: Train and evaluate the classifier on annotated data.
    * ``predict_image()``: Predict a label mask for a new image and apply postprocessing.

    Example
    -------
    ::

        from sklearn.ensemble import RandomForestClassifier

        # Initialize the processor with the desired configuration
        processor = SegmentationProcessor(
            n_neighbors=8,
            classifier=RandomForestClassifier(n_estimators=200, random_state=0),
            classes=["class1", "class2", "class3"]
        )

        # Configure preprocessing
        processor.preprocessing.set_colorspace("HSV").set_gaussian_sigma(sigma=1.5)

        # Configure postprocessing
        processor.postprocessing.set_majority_filter(numpy.ones((5, 5), dtype=bool))
        processor.postprocessing.set_background_color((10, 10, 10))

        # Train
        processor.evaluate_classifier(images_path, df, do_train=True)

        # Predict
        results = processor.predict_image(rgb_image, roi_mask)

    Parameters
    ----------
    n_neighbors : int
        Number of neighboring pixels for feature extraction.

    classifier : ClassifierMixin
        A scikit-learn classifier instance (e.g., `RandomForestClassifier()`).
        Must support predicting probabilities via `predict_proba()` method for refinement strategies.

    classes : List[str]
        Segmentation class names. Position defines the classifier integer
        label index:
        ::
            classes[0] --> 0,
            classes[1] --> 1,
            ...
            classes[N] --> N

    colors : List[Sequence[int]]
        Segmentation RGB colors (0–255 range), one per class.
        Must have the same length as ``classes``.
    """

    _SUPPORTED_REFINE_METHODS = {"bayes", "crf"}

    def __init__(
            self,
            n_neighbors: int,
            classifier: ClassifierMixin,
            classes: List[str],
            colors: Optional[List[Sequence[int]]] = None,
    ):
        # Check for probability prediction support in the classifier
        if not hasattr(classifier, "predict_proba"):
            raise ValueError(
                "Classifier must support probability predictions ('predict_proba()') "
                "for refinement strategies."
            )

        # Create a default color palette
        if colors is None:
            colors = self._create_color_palette(num_classes=len(classes))
        # Check input colors
        elif len(colors) != len(classes):
            raise ValueError(f"Number of colors ({len(colors)}) and classes ({len(classes)}) missmatch")

        # Classifier -------------------------------------
        self.class_names: List[str] = list(classes)
        self.classifier: ClassifierMixin = clone(classifier)
        self.training_metadata: Optional[dict] = None

        # Feature extractor ------------------------------
        self.feature_extractor: NeighborhoodExtractor = NeighborhoodExtractor(
            n_neighbors=n_neighbors,
            include_center=True,
            center_loc="middle"
        )

        # Processing blocks -----------------------------
        self.preprocessing = PreprocessingBlock()
        self.postprocessing = PostprocessingBlock(colors=colors)

    @property
    def n_neighbors(self) -> int:
        """Number of neighboring pixels used for feature extraction."""
        return self.feature_extractor.n_neighbors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_fitted_classifier(self):
        # check internal state
        invalid_msg = "Classifier is not fitted. Call 'evaluate_classifier()' with 'do_train=True' first."
        if self.training_metadata is None:
            raise ValueError(invalid_msg)

        # Check sklearn utility
        try:
            sklearn.utils.validation.check_is_fitted(self.classifier)
        except sklearn.exceptions.NotFittedError as e:
            raise ValueError(invalid_msg) from e

    @staticmethod
    def _create_color_palette(num_classes: int) -> List[Sequence[int]]:
        """Generate a default color palette for the given number of classes."""
        logger.debug("[Start] _create_color_palette")

        # Colormap and indices for color generation
        cmap = plt.get_cmap(name="tab10")
        indices = numpy.arange(num_classes) % 10

        # Get RGB values in 0-255 range
        colors = cmap(indices, bytes=True)[:, :3].astype(numpy.uint8)

        logger.debug("[Finish] _create_color_palette")
        return [tuple(color) for color in colors]

    def _build_feature_matrix(
        self,
        images_path: Path,
        annotations_data: pandas.DataFrame
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Load annotated images, preprocess them via the preprocessing block,
        extract neighborhood features, and return ``(X, y)``.
        """
        all_features: List[numpy.ndarray] = []
        all_labels: List[numpy.ndarray] = []

        for img_filename, img_group in annotations_data.groupby("ImageFile"):
            # Load and preprocess image
            rgb_image = skimage.io.imread(images_path / img_filename)
            image = self.preprocessing.transform(rgb_image)

            # Extract features
            yx_coordinate = img_group.loc[:, ["Cy", "Cx"]].to_numpy(dtype=int)
            features = self.feature_extractor.transform(image=image, yx_coordinates=yx_coordinate)

            # Get target labels
            labels = img_group["Class"].map(lambda cls: self.class_names.index(cls)).to_numpy(dtype=int)

            all_features.append(features)
            all_labels.append(labels)

        return numpy.vstack(all_features), numpy.concatenate(all_labels)

    @staticmethod
    def _refine_bayes(predicted_proba: numpy.ndarray, prior_proba: numpy.ndarray) -> numpy.ndarray:
        """Bayesian refinement of predicted probabilities"""
        logger.debug("[Start] _refine_bayes")

        # Validate probabilities
        if prior_proba is None:
            raise ValueError("Prior probabilities must be provided for Bayesian refinement.")
        if prior_proba.shape != predicted_proba.shape:
            raise ValueError(
                f"Prior probabilities shape {prior_proba.shape} does not match "
                f"predicted probabilities shape {predicted_proba.shape}."
            )
        if not numpy.all((prior_proba >= 0) & (prior_proba <= 1)):
            raise ValueError("Prior probabilities must be in the range [0, 1].")
        if not numpy.allclose(prior_proba.sum(axis=-1), 1.0):
            raise ValueError("Prior probabilities must sum to 1 across the last dimension (classes).")

        # Apply Bayes' theorem: P(class|data) ∝ P(data|class) * P(class)
        refined_proba = predicted_proba * prior_proba

        # Normalize the refined probabilities to ensure they sum to 1 across classes
        norm_factor = refined_proba.sum(axis=-1, keepdims=True)

        # Set uniform distribution if the norm factor is zero to avoid division by zero
        n_classes = predicted_proba.shape[-1]
        refined_proba = numpy.where(
            norm_factor > 0,
            refined_proba / norm_factor,    # Normalize to sum to 1
            1.0 / n_classes                 # If norm_factor is zero, assign uniform probabilities
        )

        logger.debug("[Finish] _refine_bayes")
        return refined_proba

    @staticmethod
    def _refine_crf(
        predicted_proba: numpy.ndarray,
        rgb_image: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        Dense CRF inference over the full image grid.

        Notes
        -----
        *   ``sxy`` (distance in pixels): control the spatial smoothness (the smaller, the more local
            the smoothing). This kernel solely looks at the spatial distance between pixels to remove
            small, isolated incorrectly-labeled regions.

        *   ``srgb`` (distance in RGB color intensity space): control the color similarity (the smaller,
            the more likely to smooth across similar colors). This kernel connects pixels that are both
            spatially nearby and have similar colors, based on the observation that nearby pixels of
            similar color usually belong to the same object class.

         *  ``compat``: control the strength of each pairwise term. Higher values lead to stronger
            smoothing effects. When the model connects two pixels using the kernels (``PairwiseGaussian``
            ``PairwiseBilateral``), this parameter introduces a penalty if those two pixels are assigned
            different labels. Can take three different types:

                *   A single number (``PottsCompatibility``): This is a simple penalty that penalizes any pair of
                    nearby/similar pixels that are given different labels equally.

                *   A 1D array (``DiagonalCompatibility``): An array of length `n_classes` with a float32
                    datatype, which allows you to apply different penalties depending on the specific label.

                *   A 2D array (``MatrixCompatibility``): An `n_classes x n_classes` matrix with a float32
                    datatype. This allows you to define a general symmetric compatibility matrix that takes
                    interactions between specific labels into account. For example, you can configure it so
                    that mistaking a "bird" pixel for "sky" is penalized less harshly than mistaking a
                    "cat" pixel for "sky". The matrix should be symmetric (μ(a,b)=μ(b,a)). The penalty for
                    Class A touching Class B should be the same as Class B touching Class A.

            Example::

                    [
                        [  0, 10, 5 ],
                        [ 10,  0, 2 ],
                        [  5,  2, 0 ]
                    ]

            Classes 0 and 1 are penalized with a strength of 10 when they touch, while classes 0 and 2
            are penalized with a strength of 5, and classes 1 and 2 are penalized with a strength of 2.
            The diagonal elements are zero because there is no penalty for pixels of the same class
            touching each other.


         *  ``kernel``: using a diagonal kernel means that the pairwise potentials only penalize
            label differences, without considering specific label interactions. This is a common choice
            for multi-class segmentation when no prior knowledge about class relationships is available.

         *  ``normalization``: symmetric normalization ensures that the influence of neighboring pixels
            is balanced and does not depend on their degree of connectivity, which can help prevent
            over-smoothing in densely connected regions.
        """
        logger.debug("[Start] _refine_crf")

        height, width = rgb_image.shape[:2]
        n_classes = predicted_proba.shape[-1]

        # Convert probabilities to the shape (num_classes, height, width)
        softmax_arr = predicted_proba.transpose(2, 0, 1).astype(numpy.float32)

        # Make arrays as contiguous in memory for pydensecrf
        softmax_arr = numpy.ascontiguousarray(softmax_arr)
        rgb_image = numpy.ascontiguousarray(rgb_image)

        # Initialize DenseCRF model and set unary potentials
        dcrf_model = dcrf.DenseCRF2D(width, height, n_classes)
        dcrf_model.setUnaryEnergy(unary_from_softmax(softmax_arr))

        # Color-independent term, features are the locations only
        # Defaults to `kernel=dcrf.DIAG_KERNEL` and `normalization=dcrf.NORMALIZE_SYMMETRIC`
        dcrf_model.addPairwiseGaussian(sxy=3, compat=3)

        # Color-dependent term, i.e. features are (x, y, r, g, b)
        # Defaults to `kernel=dcrf.DIAG_KERNEL` and `normalization=dcrf.NORMALIZE_SYMMETRIC`
        dcrf_model.addPairwiseBilateral(sxy=80, srgb=13, rgbim=rgb_image, compat=10)

        # Apply CRF inference
        # Shape (n_classes, width*height)
        refined_proba = numpy.asarray(dcrf_model.inference(15))

        # Reconstruct back to (height, width, n_classes)
        refined_proba = refined_proba.reshape((n_classes, height, width))
        refined_proba = refined_proba.transpose(1, 2, 0)

        logger.debug("[Finish] _refine_crf")
        return refined_proba

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_classifier(
        self,
        images_path: Path,
        annotations_data: pandas.DataFrame,
        do_train: bool = False
    ) -> dict:
        """
        Evaluate the classifier on annotated data and optionally train it.

        Parameters
        ----------
        images_path : Path
            Directory containing the images referenced in ``annotations_data``.

        annotations_data : pandas.DataFrame
            Data with required columns:

            *   ``"ImageFile"``: image file name (relative to ``images_path``)
            *   ``"Cx"``: x-coordinate of the annotated pixel
            *   ``"Cy"``: y-coordinate of the annotated pixel
            *   ``"Class"``: class name of the annotated pixel (must be one of the class names
                defined in the processor)

        do_train : bool, optional
            If ``True``, fit the classifier before evaluating. Default is ``False``.
            Training metadata is populated in the processor internal state.

        Returns
        -------
        dict
            Evaluation results containing the following keys:

            *   ``date_UTC``: evaluation timestamp in UTC (ISO format)

            *   ``data``: summary of the input data with keys:

                *   ``n_samples``: number of annotated samples (pixels)
                *   ``n_features``: number of features per sample (depends on the neighborhood size)
                *   ``label_counts``: dictionary with the count of samples per class label

            *   ``timings``: timing information for each step (feature extraction, training, prediction, total)

            *   ``metrics``: classification report as returned by `sklearn.metrics.classification_report()`.
        """
        logger.debug("[Start] evaluate_classifier()")

        # Check input annotations
        required_columns = {"ImageFile", "Cx", "Cy", "Class"}
        if not required_columns.issubset(annotations_data.columns):
            raise ValueError(
                f"Annotations data must contain the following columns: {required_columns}. "
                f"Got: {annotations_data.columns}"
            )
        if not numpy.issubdtype(annotations_data["Cx"].dtype, numpy.integer):
            raise ValueError(f"Column 'Cx' must be integer. Got: {annotations_data['Cx'].dtype}")
        if not numpy.issubdtype(annotations_data["Cy"].dtype, numpy.integer):
            raise ValueError(f"Column 'Cy' must be integer. Got: {annotations_data['Cy'].dtype}")
        if not set(annotations_data["Class"].unique()).issubset(set(self.class_names)):
            raise ValueError(
                "Column 'Class' contains class names that are not defined in the processor. "
                f"Expected class names: {self.class_names}. "
                f"Got: {annotations_data['Class'].unique()}"
            )

        # Check classifier state
        if not do_train:
            self._check_fitted_classifier()

        step_timings: Dict[str, float] = {}
        time_digits = 4
        t_start = time.perf_counter()

        # Feature space
        t0 = time.perf_counter()
        X, y = self._build_feature_matrix(images_path, annotations_data)
        step_timings["features"] = round(time.perf_counter() - t0, time_digits)

        # Training
        if do_train:
            t0 = time.perf_counter()
            self.classifier.fit(X, y)
            step_timings["fit"] = round(time.perf_counter() - t0, time_digits)

        # Prediction
        t0 = time.perf_counter()
        y_pred = self.classifier.predict(X)
        step_timings["predict"] = round(time.perf_counter() - t0, time_digits)
        step_timings["total"] = round(time.perf_counter() - t_start, time_digits)

        # Evaluation
        # `zero_division=numpy.nan` prevents empty classes from skewing reported averages.
        class_report = sklearn.metrics.classification_report(
            y_true=y,
            y_pred=y_pred,
            labels=range(len(self.class_names)),
            target_names=self.class_names,
            output_dict=True,
            zero_division=numpy.nan
        )

        # Data summary
        y_unique, y_counts = numpy.unique(y, return_counts=True)
        results = {
            "date_UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "data": {
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "label_counts": dict(zip(y_unique.tolist(), y_counts.tolist()))
            },
            "timings": step_timings,
            "metrics": class_report
        }

        # Populate training metadata
        if do_train:
            self.training_metadata = results

        logger.debug("[Finish] evaluate_classifier()")
        return results

    def predict_image(
        self,
        rgb_image: numpy.ndarray,
        roi_mask: Optional[numpy.ndarray] = None,
        class_priors: Optional[numpy.ndarray] = None,
        refine: Optional[str] = None,
        save_file: Optional[Path] = None
    ) -> dict:
        """
        Segment an image using the trained classifier.

        Parameters
        ----------
        rgb_image : numpy.ndarray
            Input RGB image of shape `(height, width, 3)` and `uint8` data type.

        roi_mask : numpy.ndarray, optional
            Boolean mask of shape `(height, width)`. Only `True` pixels are classified.
            For pixels outside the valid region (`False` pixels), output results behave
            as follows:

            * labels: receive the postprocessing ignore index value.
            * color_labels: receive the postprocessing background color.
            * probabilities: receive a uniform distribution over classes (1 / n_classes).

        class_priors : numpy.ndarray, optional
            Spatial prior probabilities of shape `(height, width, n_classes)`.
            Required if `refine="bayes"`.

        refine : str, optional
            Predicted probabilities refinement method to apply. If `None` (default), no refinement
            is applied. Supported methods:

            * ``"bayes"``: Bayesian refinement using the provided `class_priors`.
            * ``"crf"``: Dense CRF refinement using the input RGB image for pairwise potentials.

        save_file : Path, optional
            If provided, saves the predicted results as a compressed NPZ file at the specified path.
            The NPZ file contains the same data as the returned dictionary.

        Returns
        -------
        dict
            Predicted results data. See ``roi_mask`` parameter for behavior of each output
            in masked-out areas.

            *   ``"classes"``: List of class names corresponding to the predicted labels.
            *   ``"image"``: The original input RGB image.
            *   ``"class_proba"``: predicted class probabilities. 3D array as `(height, width, n_classes)`
            *   ``"labels"``: predicted class labels. 2D integer array as `(height, width)`.
            *   ``"color_labels"``: color-coded image labels. 3D uint8 RGB array as `(height, width, 3)`.
            *   ``"overlay"``: overlay visualization. 3D uint8 RGB array of shape `(height, width, 3)`.
        """
        logger.debug("[Start] predict_image")

        self._check_fitted_classifier()
        rgb_image = skimage.util.img_as_ubyte(rgb_image)
        height, width = rgb_image.shape[:2]
        n_classes = len(self.class_names)

        # Preprocessing block
        image = self.preprocessing.transform(rgb_image)

        # Model Inference
        # ------------------------------------------------------------------------
        # Feature extraction
        # Shape (n_samples, n_features)
        X = self.feature_extractor.transform(image=image, mask=roi_mask)

        # Raw predicted probabilities
        # Shape (n_samples, n_classes)
        predicted_proba = self.classifier.predict_proba(X)

        # Full predicted probabilities with uniform probability for unclassified pixels
        # Shape (height, width, n_classes)
        if roi_mask is not None:
            dummy_proba = numpy.full(
                shape=(height, width, n_classes),
                fill_value=1.0 / n_classes,
                dtype=numpy.float32
            )
            dummy_proba[roi_mask] = predicted_proba
            predicted_proba = dummy_proba

        # Refined probabilities (optional)
        if refine is not None:
            if refine not in self._SUPPORTED_REFINE_METHODS:
                raise ValueError(
                    f"Unsupported refinement method: '{refine}'. "
                    f"Supported methods: {self._SUPPORTED_REFINE_METHODS}"
                )
            elif refine == "bayes":
                predicted_proba = self._refine_bayes(predicted_proba=predicted_proba, prior_proba=class_priors)
            elif refine == "crf":
                predicted_proba = self._refine_crf(predicted_proba=predicted_proba, rgb_image=rgb_image)
        # ------------------------------------------------------------------------

        # Output results
        final_results = {
            "classes": self.class_names,
            "image": rgb_image,
            "class_proba": predicted_proba,
        }

        # Postprocessing block
        postprocessed_results = self.postprocessing.apply(
            image=rgb_image,
            predicted_proba=predicted_proba,
            roi_mask=roi_mask
        )
        final_results.update(postprocessed_results)

        # Persistence as NPZ (optional)
        if save_file is not None:
            save_file.parent.mkdir(parents=True, exist_ok=True)
            numpy.savez_compressed(save_file.with_suffix(".npz"), **final_results)
            logger.info(f"Predicted results saved: '{save_file}'")

        logger.debug("[Finish] predict_image")
        return final_results

    def to_pkl(self, filepath: Path):
        """Serialize this processor to a compressed joblib PKL file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Processor saved: '{filepath}'")

    def get_metadata(self) -> dict:
        """JSON-serializable metadata dictionary."""
        processor_blocks = {
            "preprocessing": self.preprocessing.to_json(),
            "feature_extraction": self.feature_extractor.to_json(),
            "classifier": {
                "type": type(self.classifier).__name__,
                "framework": {"name": "scikit-learn", "version": sklearn.__version__},
                "hyperparameters": self.classifier.get_params(),
            },
            "postprocessing": self.postprocessing.to_json(),
        }

        return {
            "class_names": self.class_names,
            "n_neighbors": self.n_neighbors,
            "blocks": processor_blocks,
            "training": self.training_metadata
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  classes={self.class_names},\n"
            f"  n_neighbors={self.n_neighbors},\n"
            f"  preprocessing={self.preprocessing!r},\n"
            f"  postprocessing={self.postprocessing!r}\n"
            f")"
        )

