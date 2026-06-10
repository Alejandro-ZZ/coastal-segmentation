import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import joblib
import matplotlib.pyplot as plt
import numpy
import pandas
import skimage
from skimage.util import img_as_uint
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier


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

    def validate_input(
            self,
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
                    f"Row (Y) coordinates must be in the range [0, {height - 1}]. "
                    f"Got: [{y_coords.min()}, {y_coords.max()}]"
                )
            if (x_coords < 0).any() or (x_coords >= width).any():
                raise ValueError(
                    f"Column (X) coordinates must be in the range [0, {width - 1}]. "
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
            If not provided and `yx_coordinates` is None, all image pixels are computed.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape `(n_samples, n_features)`

            *   ``n_samples`` refers to the number of pixel coordinates in the input ``yx_coordinates`` or
                defined by the ``mask``. If both are None, it corresponds to the total number of pixels in
                the input image, i.e., `height * width`.

            *   ``n_features`` corresponds to the number of neighborhood features extracted for each pixel
                coordinate. It is defined as: `image_channels * (n_neighbors + 1)`. Values are ordered
                sequentially with the center pixel being in the middle of the array followed by its neighbors
                in incremental order: 1, 2, 3, ... , `n_neighbors`.
        """
        # ---------------------
        # INPUT VALIDATION
        # ---------------------
        if isinstance(yx_coordinates, list):
            yx_coordinates = numpy.asarray(yx_coordinates)
        self.validate_input(image=image, yx_coordinates=yx_coordinates, mask=mask)

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

    def apply(self, predicted_proba: numpy.ndarray, roi_mask: Optional[numpy.ndarray] = None) -> dict:
        """
        Apply postprocessing to an assembled 2-D label mask.

        Steps
        -----
        1. **Majority filter** (optional): replace each pixel with the most
           common label in its neighbourhood, only inside the valid region.
        2. **Color visualization**: map integer labels to RGB colors.

        Parameters
        ----------
        predicted_proba : numpy.ndarray
            3-D array of shape (height, width, num_classes) containing the predicted class
            probabilities for each pixel. Values must be in the range [0, 1] and sum to 1
            across the last dimension (classes).

        roi_mask : numpy.ndarray, optional
            Boolean 2-D mask ``(H, W)``. When provided:

            * The majority filter is applied only inside the valid region.
            * ``ignore_index`` pixels receive ``_bg_color`` in the visualization.

        Returns
        -------
        dict
            * ``"labels"``       – refined 2-D ``int32`` array ``(H, W)``.
            * ``"color_labels"`` – 3-D ``uint8`` RGB array ``(H, W, 3)``.
        """
        logger.debug(f"[Start] {self.__class__.__name__}.apply()")

        # Check predicted probabilities
        self._check_predicted_proba(predicted_proba)

        # Predicted integer labels
        predicted_labels = numpy.argmax(predicted_proba, axis=-1).astype(numpy.uint32)
        if roi_mask is not None:
            predicted_labels[~roi_mask] = self.ignore_index

        # Majority filter (optional)
        if self._majority_footprint is not None:
            predicted_labels = self._apply_majority_filter(predicted_labels, roi_mask)

        # Color visualization
        all_colors = self.colors + [self._bg_color]
        color_palette = numpy.array(all_colors, dtype=numpy.uint8)
        color_labels = color_palette[predicted_labels]

        # TODO: Overlay image ("overlay")

        logger.debug(f"[Finish] {self.__class__.__name__}.apply()")
        return {
            "labels": predicted_labels,
            "color_labels": color_labels
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

    * ``preprocessing``  – :class:`PreprocessingBlock`  (colorspace, Gaussian blur)
    * ``postprocessing`` – :class:`PostprocessingBlock` (majority filter, visualization)
    * ``classifier``    – ``RandomForestClassifier``   (owned directly by the processor)

    Construct with the minimum required arguments, then configure each block via
    its builder interface::

        processor = SegmentationProcessor(
            classes=["Water", "Sand", "Vegetation"],
            colors=[[0, 0, 255], [255, 255, 0], [0, 128, 0]]
        )

        # Configure preprocessing
        processor.preprocessing \\
            .set_colorspace("HSV") \\
            .set_gaussian_sigma(sigma=1.5)

        # Configure postprocessing
        processor.postprocessing \\
            .set_majority_filter(numpy.ones((5, 5), dtype=bool)) \\
            .set_background((10, 10, 10))

        # Configure feature extraction and classifier (processor-level)
        processor.set_neighbors(8).set_classifier(n_estimators=200, random_state=0)

        # Train
        processor.evaluate_classifier(images_path, df, do_train=True)

        # Predict
        result = processor.predict_image(rgb_image, roi_mask=mask)

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
    @staticmethod
    def _create_color_palette(num_classes: int) -> List[Sequence[int]]:
        """Generate a default color palette for the given number of classes."""
        # Colormap and indices for color generation
        cmap = plt.get_cmap(name="tab10")
        indices = numpy.arange(num_classes) % 10

        # Get RGB values in 0-255 range
        colors = cmap(indices, bytes=True)[:, :3].astype(numpy.uint8)
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
        import skimage.io

        all_features: List[numpy.ndarray] = []
        all_labels: List[numpy.ndarray] = []

        for image_file_name, group in annotations_data.groupby("ImageFile"):
            rgb_image = skimage.io.imread(images_path / image_file_name)

            # Delegate image preprocessing to the block
            image = self.preprocessing.transform(rgb_image)

            cx = group["Cx"].values.astype(int)
            cy = group["Cy"].values.astype(int)
            yx_coords = numpy.stack((cy, cx), axis=1)

            features = self.feature_extractor.transform(image=image, yx_coordinates=yx_coords)
            labels = numpy.array(
                [self.class_names.index(cls) for cls in group["Class"]],
                dtype=int
            )
            all_features.append(features)
            all_labels.append(labels)

        return numpy.vstack(all_features), numpy.concatenate(all_labels)

    def _assemble_label_mask(
        self,
        predicted_labels: numpy.ndarray,
        yx_coords: numpy.ndarray,
        height: int,
        width: int,
        roi_mask: Optional[numpy.ndarray]
    ) -> numpy.ndarray:
        """
        Scatter a sparse 1-D ``predicted_labels`` array into a full ``(H, W)``
        ``int32`` label mask. Pixels not present in ``yx_coords`` receive
        ``postprocessing.ignore_index``.
        """
        ignore_index = self.postprocessing.ignore_index
        if roi_mask is not None:
            label_mask = numpy.full(
                (height, width), fill_value=ignore_index, dtype=numpy.int32
            )
        else:
            label_mask = numpy.empty((height, width), dtype=numpy.int32)
        label_mask[yx_coords[:, 0], yx_coords[:, 1]] = predicted_labels
        return label_mask

    def _predict(
        self,
        X: numpy.ndarray,
        rgb_image: numpy.ndarray,
        yx_coords: numpy.ndarray,
        class_priors: Optional[numpy.ndarray],
        refine: Optional[str],
        height: int,
        width: int
    ) -> numpy.ndarray:
        """
        Classify pixels in ``X``, applying the chosen refinement strategy.

        Refinement strategies
        ---------------------
        * ``None``      – raw RF ``predict()``.
        * ``"bayes"``   – RF posteriors multiplied by spatial ``class_priors``, re-normalised.
        * ``"crf"``     – dense CRF inference (requires ``pydensecrf``).
        """
        if refine == "bayes" and class_priors is not None:
            posteriors = self.classifier.predict_proba(X)
            pixel_priors = class_priors[yx_coords[:, 0], yx_coords[:, 1], :]
            posteriors = posteriors * pixel_priors
            row_sums = posteriors.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0   # guard against all-zero prior
            posteriors /= row_sums
            return numpy.argmax(posteriors, axis=1)

        elif refine == "crf":
            return self._refine_crf(
                rgb_image=rgb_image,
                yx_coords=yx_coords,
                X=X,
                height=height,
                width=width
            )

        else:
            return self.classifier.predict(X)

    def _refine_crf(
        self,
        rgb_image: numpy.ndarray,
        yx_coords: numpy.ndarray,
        X: numpy.ndarray,
        height: int,
        width: int
    ) -> numpy.ndarray:
        """Dense CRF inference over the full image grid. Requires ``pydensecrf``."""
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
        except ImportError:
            raise ImportError(
                "CRF refinement requires 'pydensecrf'. "
                "Install it with: pip install pydensecrf"
            )

        n_classes = len(self.class_names)
        proba = self.classifier.predict_proba(X)
        prob_volume = numpy.zeros((height, width, n_classes), dtype=numpy.float32)
        prob_volume[yx_coords[:, 0], yx_coords[:, 1]] = proba
        prob_chw = numpy.clip(prob_volume.transpose(2, 0, 1), 1e-6, 1.0)

        d = dcrf.DenseCRF2D(width, height, n_classes)
        d.setUnaryEnergy(unary_from_softmax(prob_chw))
        d.addPairwiseGaussian(sxy=3, compat=3)

        img_u8 = (
            rgb_image if rgb_image.dtype == numpy.uint8
            else (rgb_image * 255).astype(numpy.uint8)
        )
        d.addPairwiseBilateral(sxy=40, srgb=10, rgbim=img_u8, compat=10)

        Q = numpy.array(d.inference(5)).reshape(n_classes, height, width)
        full_map = numpy.argmax(Q, axis=0)
        return full_map[yx_coords[:, 0], yx_coords[:, 1]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_classifier(
        self,
        images_path: Path,
        annotations_data: Any,   # pandas.DataFrame
        do_train: bool = False
    ) -> dict:
        """
        Optionally train the classifier and evaluate it on annotated pixel data.

        Parameters
        ----------
        images_path : Path
            Directory containing the images referenced in ``annotations_data``.

        annotations_data : pandas.DataFrame
            DataFrame with columns: ``"ImageFile"``, ``"Cx"``, ``"Cy"``, ``"Class"``.

        do_train : bool, optional
            If ``True``, fit the classifier before evaluating. Default is ``False``.

        Returns
        -------
        dict
            * ``"data"``    – ``{"n_samples", "n_features", "labels"}``
            * ``"timings"`` – ``{"features", "fit"`` (only when ``do_train=True``), ``"predict", "total"}``
            * ``"metrics"`` – ``{"class_report": <sklearn classification_report dict>}``
        """
        import time
        from datetime import datetime, timezone
        from sklearn.metrics import classification_report

        t_start = time.perf_counter()

        t0 = time.perf_counter()
        X, y = self._build_feature_matrix(images_path, annotations_data)
        timings: Dict[str, float] = {"features": round(time.perf_counter() - t0, 4)}

        if do_train:
            t0 = time.perf_counter()
            self.classifier.fit(X, y)
            timings["fit"] = round(time.perf_counter() - t0, 4)

        t0 = time.perf_counter()
        y_pred = self.classifier.predict(X)
        timings["predict"] = round(time.perf_counter() - t0, 4)
        timings["total"] = round(time.perf_counter() - t_start, 4)

        class_report = classification_report(
            y_true=y,
            y_pred=y_pred,
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        results = {
            "data": {
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "labels": sorted(int(v) for v in numpy.unique(y))
            },
            "timings": timings,
            "metrics": {"class_report": class_report}
        }

        if do_train:
            self.training_metadata = {
                "date_UTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
                **results
            }

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

        Pipeline
        --------
        1. ``preprocessing.transform(rgb_image)``     – colorspace + blur
        2. Feature extraction via ``feature_extractor``
        3. Classification + optional refinement (``_predict``)
        4. Assemble full-image label mask (``_assemble_label_mask``)
        5. ``postprocessing.apply(label_mask, roi_mask)`` – filter + colorize

        Parameters
        ----------
        rgb_image : numpy.ndarray
            Input RGB image ``(H, W, 3)``, ``uint8`` or ``float``.

        roi_mask : numpy.ndarray, optional
            Boolean mask ``(H, W)``. Only ``True`` pixels are classified.

        class_priors : numpy.ndarray, optional
            Spatial prior probabilities ``(H, W, n_classes)``.
            Required when ``refine="bayes"``.

        refine : str, optional
            ``None`` (raw RF), ``"bayes"``, or ``"crf"``.

        save_file : Path, optional
            If provided, saves the integer label array as a compressed NPZ
            (key ``"labels"``) at this path.

        Returns
        -------
        dict
            * ``"labels"``       – 2-D ``int32`` array ``(H, W)``
            * ``"color_labels"`` – 3-D ``uint8`` RGB array ``(H, W, 3)``
        """
        height, width = rgb_image.shape[:2]

        # ── 1. Preprocessing block ───────────────────────────────────────
        image = self.preprocessing.transform(rgb_image)

        # ── 2. Determine pixels to classify ─────────────────────────────
        if roi_mask is not None:
            yx_coords = numpy.argwhere(roi_mask)
        else:
            rows, cols = numpy.meshgrid(
                numpy.arange(height), numpy.arange(width), indexing="ij"
            )
            yx_coords = numpy.stack((rows.ravel(), cols.ravel()), axis=1)

        # ── 3. Feature extraction ────────────────────────────────────────
        X = self.feature_extractor.transform(image=image, yx_coordinates=yx_coords)

        # ── 4. Prediction + refinement ───────────────────────────────────
        predicted_labels = self._predict(
            X=X,
            rgb_image=rgb_image,
            yx_coords=yx_coords,
            class_priors=class_priors,
            refine=refine,
            height=height,
            width=width
        )

        # ── 5. Assemble full-image label mask ────────────────────────────
        label_mask = self._assemble_label_mask(
            predicted_labels=predicted_labels,
            yx_coords=yx_coords,
            height=height,
            width=width,
            roi_mask=roi_mask
        )

        # ── 6. Postprocessing block ──────────────────────────────────────
        result = self.postprocessing.apply(predicted_proba=label_mask, roi_mask=roi_mask)

        # ── 7. Optional persistence ──────────────────────────────────────
        if save_file is not None:
            save_file = Path(save_file)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            numpy.savez_compressed(save_file, labels=result["labels"])

        return result

    def to_pkl(self, filepath: Path) -> None:
        """Serialize this processor to a compressed joblib PKL file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"SegmentationProcessor saved: {filepath}")

    def get_metadata(self) -> dict:
        """
        Return a JSON-serializable metadata dictionary.

        Returns
        -------
        dict
            * ``"preprocessing"``      – config from :class:`PreprocessingBlock`.
            * ``"postprocessing"``     – config from :class:`PostprocessingBlock`.
            * ``"feature_extraction"`` – ``{"n_neighbors"}``.
            * ``"classifier"``         – type, scikit-learn version, hyperparameters.
            * ``"training"``           – present only after ``evaluate_classifier(do_train=True)``.
        """
        import sklearn

        metadata: Dict[str, Any] = {
            "preprocessing": self.preprocessing.get_config(),
            "postprocessing": self.postprocessing.get_config(),
            "feature_extraction": {
                "n_neighbors": self.n_neighbors,
            },
            "classifier": {
                "type": type(self.classifier).__name__,
                "framework": {
                    "name": "scikit-learn",
                    "version": sklearn.__version__,
                },
                "hyperparameters": self.classifier.get_params(),
            },
        }

        if self.training_metadata is not None:
            metadata["training"] = self.training_metadata

        return metadata

    def __repr__(self) -> str:
        return (
            f"SegmentationProcessor(\n"
            f"  classes={self.class_names},\n"
            f"  n_neighbors={self.n_neighbors},\n"
            f"  preprocessing={self.preprocessing!r},\n"
            f"  postprocessing={self.postprocessing!r}\n"
            f")"
        )

