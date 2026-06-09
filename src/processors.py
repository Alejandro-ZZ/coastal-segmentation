import logging
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy


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


