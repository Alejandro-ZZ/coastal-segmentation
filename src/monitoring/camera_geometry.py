import cv2
import logging
import matplotlib.patches as mpatches
import numpy
import scipy

from numpy.typing import NDArray
from pathlib import Path 
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union
)


logger = logging.getLogger("CameraGeometry")


class CameraGeometry:
    """
    Stores the geometric properties of a camera:
    
    1. Calibration parameters for a camera at one native image size.
    2. Rectification parameters for transforming the image to a rectified view.

    Parameters
    ----------
    bbox : Optional[tuple], optional
        Bounding box for the rectified image. Tuple of (X_min, Y_min, Width, Height). 
        Where `X_min` and `Y_min` are the minimum world coordinates, and `Width` and 
        `Height` are the dimensions of the rectified image in world units (e.g., meters).
        
        If None (default), try to use the full image.
    
    resolution : float, optional
        Pixel size for the rectified output grid in world units (e.g., meters). Default is 1.0.
    """
    def __init__(self, bbox: Optional[tuple] = None, resolution: float = 1.0):
        if bbox is None:
            logger.warning("Currently computing issues are expected when no bounding box is provided.")
        
        # Calibration parameters
        self._camera_mtx: NDArray = numpy.array([])
        self._dist_coeffs: NDArray = numpy.array([])
        self._calib_meta: Dict[str, Any] = {}

        # Rectification parameters
        self._bounding_box: Optional[tuple] = bbox
        self._xy_resolution: tuple = (resolution, resolution)
        self._homography_mtx: NDArray = numpy.array([])
        self._homography_meta: Dict[str, Any] = {}


    # CAMERA PARAMETERS
    # -----------------------------
    def calibration_params(self, serialize: bool = False) -> Dict[str, Any]:
        """
        Camera calibration parameters and metadata.
        
        Parameters
        ----------
        serialize : bool, optional
            If True, returns the parameters in a serializable format (e.g., lists instead of numpy arrays).
            Default is False.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:

            -   ``matrix``: Intrinsic matrix. 2D-FloatArray of shape (3, 3).
            -   ``coefficients``: Distortion coefficients. 2D-FloatArray of shape (1, N) | N = {4, 5, 8, 12, 14}.
            -   ``metadata``: Additional calibration metadata.
        """
        if serialize:
            return {
                "matrix": self._camera_mtx.tolist(),
                "coefficients": self._dist_coeffs.tolist(),
                "metadata": self._calib_meta
            }
        else:
            return {
                "matrix": self._camera_mtx,
                "coefficients": self._dist_coeffs,
                "metadata": self._calib_meta
            }

    def rectification_params(self, serialize: bool = False) -> Dict[str, Any]:
        """
        Camera rectification parameters and metadata.

        Parameters
        ----------
        serialize : bool, optional
            If True, returns the parameters in a serializable format (e.g., lists instead of numpy arrays).
            Default is False.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:

            -   ``homography``: Homography matrix to transform pixel to world coordinates. 2D-FloatArray of shape (3, 3).
            
            -   ``bbox``: Bounding box for the rectified image. Tuple of (X_min, Y_min, Width, Height). 
                Where `X_min` and `Y_min` are the minimum world coordinates, and `Width` and `Height` are the dimensions 
                of the rectified image in world units (e.g., meters).
            
            -   ``resolution``: Pixel size for the rectified output grid in world units (e.g., meters).
            
            -   ``metadata``: Additional rectification metadata.
        """
        if serialize:
            return {
                "homography": self._homography_mtx.tolist(),
                "bbox": self._bounding_box,
                "resolution": self._xy_resolution,
                **self._homography_meta
            }
        else:
            return {
                "homography": self._homography_mtx,
                "bbox": self._bounding_box,
                "resolution": self._xy_resolution,
                **self._homography_meta
        }


    # CAMERA CALIBRATION
    # -----------------------------
    def is_calibrated(self) -> bool:
        """Return True if the camera has been calibrated with chessboard images."""
        return (self._camera_mtx.size > 0) and (self._dist_coeffs.size > 0)

    def _find_chessboard_corners(
            self, 
            image: NDArray, 
            pattern_size: tuple, 
            refine: bool, 
            drawn_file: Optional[Path] = None
        ) -> Optional[NDArray]:
        """
        Find the corners of a chessboard in the given image.

        Parameters
        ----------
        image : NDArray
            The input image in which to find the chessboard corners.
        
        pattern_size : tuple
            The number of internal corners per a chessboard row and column (points_per_row, points_per_colum).
        
        refine : bool
            If True, refine the corner locations using `cv2.cornerSubPix()`.
        
        drawn_file : Path, optional
            If provided, the image with the detected corners will be saved to this file.
        
        Returns
        -------
        NDArray | None
            The detected corners as a numpy array of shape (N, 1, 2) if found, otherwise None.
        """
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        find_success, corners = cv2.findChessboardCorners(image_gray, pattern_size, corners=None)
        if find_success:
            # Refine detection if required
            if refine:
                # `winSize` is the half-size of the search window for corner refinement. The window size is
                #       `2*winSize+1`, and is centered on each corner to perform the refinement.
                # `zeroZone` defines a region around the center of the search window where the gradient (or intensity)
                #       is ignored. `zeroZone=(-1, -1)` means no region is ignored. This is usually enough and
                #       recommended in most cases.
                corners = cv2.cornerSubPix(
                    image=image_gray,
                    corners=corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )

            # If required, draw detected corners and save it
            if drawn_file is not None:
                cv2.drawChessboardCorners(image, pattern_size, corners, patternWasFound=find_success)
                cv2.imwrite(drawn_file.as_posix(), image)
        else:
            corners = None
    
        return corners

    def _calibrate_camera(
            self, 
            obj_points: List[NDArray], 
            img_points: List[NDArray], 
            image_width: int,
            image_height: int, 
            compute_error: bool = True
    ) -> Dict[str, Any]:
        """
        Calibrates the camera using the provided object and image points.

        Parameters
        ----------
        obj_points : List[NDArray]
            List of object points in the world coordinate system.

        img_points : List[NDArray]
            List of corresponding image points in the image coordinate system.
        
        image_width, image_height : int
            The width and height of the images.
        
        compute_error : bool, optional
            If True, compute the re-projection error to estimate how exact the found parameters are.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the calibration success, camera matrix, distortion coefficients, 
            and mean pixel error.
        """
        #         mtx --> camera intrinsic matrix (3x3)
        # dist_coeffs --> distortion coefficients (1xN)
        #      r_vecs --> rotation vectors (1x3) for each image
        #       t_ves --> translation vectors (1x3) for each image
        calib_success, mtx, dist_coeffs, r_vecs, t_vecs = cv2.calibrateCamera(
            objectPoints=obj_points,
            imagePoints=img_points,
            imageSize=(image_width, image_height),
            cameraMatrix=numpy.array([]),   # None
            distCoeffs=numpy.array([]),     # None
        )

        # Compute the re-projection error
        mean_error = None
        if compute_error:
            mean_error = 0
            for idx in range(len(obj_points)):
                # Re-project 3d points in real world space to 2d points in the image plane
                img_points2, _ = cv2.projectPoints(
                    objectPoints=obj_points[idx],
                    rvec=r_vecs[idx],
                    tvec=t_vecs[idx],
                    cameraMatrix=mtx,
                    distCoeffs=dist_coeffs
                )
                error = cv2.norm(src1=img_points[idx], src2=img_points2, normType=cv2.NORM_L2) / len(img_points2)
                mean_error += error
            mean_error = round(mean_error / len(obj_points), 3)

        # Output camera parameters
        return {
            "success": calib_success,
            "camera_matrix": mtx,
            "distortion_coefficients": dist_coeffs,
            "mean_pixel_error": mean_error
        }

    def calibrate_from_chessboards(
        self,
        filepaths: List[Path],
        pattern_size: Tuple[int, int],
        compute_error: bool = True,
        refine_corners: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Finds the camera intrinsic and extrinsic parameters from several views of a calibration
        chessboard pattern.

        Reference: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

        Parameters
        ----------
        filepaths : List[Path]
            List of paths to reference images with different views of the calibration
            chessboard pattern from the same camera.

        pattern_size : Tuple[int, int]
            Chessboard pattern described as the number of internal corners per a chessboard row and columns.
            E.g.: (points_per_row, points_per_colum)

        compute_error : bool, optional
            If True, compute the re-projection error to estimate how exact the found parameters are. The closer the
            re-projection error is to zero, the more accurate the found parameters are.

        refine_corners : bool, optional
            If True (default), refines the found corner locations using `cv2.cornerSubPix()`.
        
        output_path : Optional[Path], optional
            If provided, save the chessboard images with overlaid found corners in the specified path.
            Output files will be named as: <input_file_name>_corners.<input_file_extension> 

        Returns
        -------
        Dict[str, Any]
            Dictionary with the camera intrinsic and extrinsic parameters:

            -   ``camera_matrix`` (numpy.ndarray): 3x3 floating-point camera intrinsic matrix.
            
            -   ``distortion_coefficients`` (numpy.ndarray): 2D floating-point array of shape (1, N)
                where 'N' can be 4, 5, 8, 12 or 14 elements.
            
            - ``mean_pixel_error`` (float): mean re-projection pixel error. None If compute_error is False.


            If no corners were found to any chessboard file, return an empty dictionary.
        """
        logger.debug("[Start] calibrate_from_chessboard")
        if len(filepaths) == 0:
            raise ValueError("Empty chessboard pattern files.")
        if len(filepaths) < 4:
            logger.warning(
                f"Calibration may be inaccurate. Few chessboard files provided: {len(filepaths)}. "
                f"At least 4 files are recommended."
            )
        logger.debug(f"Calibrating with {len(filepaths)} chessboard pattern files")

        # Prepare output directory for drawn corners
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        # Chessboard pattern size
        points_per_row, points_per_colum = pattern_size

        # Prepare object points, like: (0,0,0), (1,0,0), (2,0,0) ...,(6,5,0)
        # The third dimension is always 0 for a planar pattern.
        objp = numpy.zeros((points_per_row * points_per_colum, 3), numpy.float32)
        objp[:, :2] = numpy.mgrid[0:points_per_colum, 0:points_per_row].T.reshape(-1, 2)

        # Lists to store object points and image points from all the images
        obj_points: List[NDArray] = []  # 3d point in real world space
        img_points: List[NDArray] = []  # 2d points in image plane

        # Get the chessboard image shape from the first file
        image_shape = cv2.imread(filepaths[0].as_posix()).shape

        # Process each chessboard pattern image
        for filepath in filepaths:
            # Read reference image as BGR and gray
            image: NDArray = cv2.imread(filepath.as_posix())

            # Warn about different image sizes. All patterns should be the same size.
            if image_shape != image.shape:
                logger.warning(
                    f"Different chessboard image sizes. Expected: {image_shape}."
                    f"Got: {image.shape}. File: '{filepath.name}'."
                )

            # Output file path to save drawn corners
            output_file = None
            if output_path is not None:
                output_file = output_path / f"{filepath.stem}_corners{filepath.suffix}"
            
            # Find the chessboard corners
            corners = self._find_chessboard_corners(image, pattern_size, refine_corners, output_file)
            if corners is not None:
                obj_points.append(objp)
                img_points.append(corners)
            else:
                logger.warning(f"No corners found. File: '{filepath.name}'")

        # Display a summary of fail to detect corners
        if len(obj_points) != len(filepaths) and len(img_points) != len(filepaths):
            logger.warning(
                f"Corners not detected in all images. Correctly detected: {len(img_points)}. "
                f"Total files: {len(filepaths)}"
            )

        # Output camera parameters 
        calib_params: Dict[str, Any] = {}
        if len(obj_points) == 0 and len(img_points) == 0:
            # No corners found for all chessboard files
            logger.error("No corners were found for any input chessboard file")
        else:
            # If any corner were detected, process them
            calib_params = self._calibrate_camera(
                obj_points=obj_points,
                img_points=img_points,
                image_width=image_shape[1],
                image_height=image_shape[0],
                compute_error=compute_error
            )

            # Populate attributes
            if calib_params.get("success", False):
                self._camera_mtx = calib_params["camera_matrix"]
                self._dist_coeffs = calib_params["distortion_coefficients"]
                self._calib_meta = {
                    "mean_pixel_error": calib_params["mean_pixel_error"],
                    "method": "chessboard",
                    "pattern_size": pattern_size,
                }
            else:
                logger.error("Camera calibration failed.")

        logger.debug("[Finish] calibrate_from_chessboard")
        return self.calibration_params()

    def compute_homography(
            self,
            image_coordinates: numpy.ndarray,
            world_coordinates: numpy.ndarray,
            z: float = 0,
            compute_error: bool = False
    ) -> Dict[str, Any]:
        """
        Computes the homography matrix for rectifying an image plane (z)
        based on ground control points (GCP's).

        Parameters
        ----------
        image_coordinates : numpy.ndarray
            2D array of shape (N, 2) containing the image GCP's in pixels.
            Coordinate points are expected to be as: (column/x, row/y).

        world_coordinates : numpy.ndarray
            2D array of shape (N, 3) containing real-world GCP's in meters.
            Coordinate points are expected to be as: (X, Y, Z).

        z : float
            The Z-coordinate (elevation) in the real-world coordinate system of the
            plane to be rectified. This plane (Z_world = z) will be the plane that
            appears fronto-parallel when projecting an image. Defaults to 0.

        compute_error : bool
            If True, compute re-projection errors in pixels.

        Returns
        -------
        dict
            Dictionary containing the homography matrix and metadata:
            -   ``homography_matrix`` (numpy.ndarray): 3x3 homography matrix
            -   ``z_plane`` (float): Z-coordinate of the projection plane
            -   ``mean_pixel_error`` (float): mean re-projection pixel error. -1.0 if compute_error is False.
            -   ``mean_meter_error`` (float): mean re-projection error in meters. -1.0 if compute_error is False.
        """
        logger.debug("[Start] compute_homography")
        
        # Check if the camera has been calibrated
        if not self.is_calibrated():
            raise RuntimeError("Camera has not been calibrated. Cannot compute homography.")
        if len(image_coordinates) < 4:
            raise ValueError("At least 4 image coordinates are required to compute homography.")

        image_coordinates = numpy.asarray(image_coordinates).astype(numpy.float32)  # column/x, row/y
        world_coordinates = numpy.asarray(world_coordinates).astype(numpy.float32)  # X, Y, Z

        # Estimate the camera pose
        #    rotation_vector --> shape: (3, 1)
        # translation_vector --> shape: (3, 1)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=world_coordinates, # 3D points (X, Y, Z)
            imagePoints=image_coordinates,  # 2D projections (column, row)
            cameraMatrix=self._camera_mtx,
            distCoeffs=self._dist_coeffs
        )

        # Convert a rotation vector (Rodrigues representation) to a rotation matrix
        # Only the first element (index 0) is used. We do not use the jacobian matrix (index 1)
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0] # shape: (3, 3)

        # Assume height of projection plane
        rotation_matrix[:, 2] = rotation_matrix[:, 2] * z

        # Add the translation vector
        rotation_matrix[:, 2] = rotation_matrix[:, 2] + translation_vector.flatten()

        # Compute homography and normalize it
        homography = numpy.linalg.inv(numpy.dot(self._camera_mtx, rotation_matrix))
        homography = homography / homography[-1, -1]

        # Populate attributes
        self._homography_mtx = homography

        # Compute re-projection erros
        if compute_error:
            # Compute error in pixels
            tot_error = 0
            total_points = 0
            for i in range(len(world_coordinates)):
                reprojected_image_points, _ = cv2.projectPoints(
                    objectPoints=world_coordinates[i],
                    rvec=rotation_vector,
                    tvec=translation_vector,
                    cameraMatrix=self._camera_mtx,
                    distCoeffs=self._dist_coeffs
                )
                tot_error += numpy.sum(numpy.abs(image_coordinates[i] - reprojected_image_points)**2)
                total_points += i
            mean_pixel_error = numpy.sqrt(tot_error / total_points)

            # Compute error in meters
            image_coordinates_undistorted = self.undistort_points(image_coordinates)
            reprojected_world_points = self.rectify_points(image_coordinates_undistorted)
            meter_errors = numpy.linalg.norm(reprojected_world_points - world_coordinates[:, :2], axis=1)
            mean_meter_error = numpy.mean(meter_errors)
            best_error_ids = numpy.argsort(meter_errors)
            logger.debug(f"Best error IDs: {best_error_ids}")

            mean_pixel_error = round(float(mean_pixel_error), 4)
            mean_meter_error = round(float(mean_meter_error), 4)
        else:
            mean_pixel_error = -1.0
            mean_meter_error = -1.0

        # Populate attributes
        self._homography_meta = {
            "z_plane": z,
            "mean_pixel_error": mean_pixel_error,
            "mean_meter_error": mean_meter_error,
            "num_points": len(world_coordinates)
        }

        logger.debug("[Finish] compute_homography")
        return self.homography_params()


    # GEOMETRIC TRANSFORMATION
    # -----------------------------
    def undistort_points(self, image_coordinates: NDArray) -> NDArray:
        """
        Undistort image coordinates based on camera parameters.

        Parameters
        ----------
        image_coordinates : numpy.ndarray
            2D array of shape Nx2 containing image coordinates in pixels of GCPs.
            Coordinates are expected to be as: (column/x, row/y).

        Returns
        -------
        points : numpy.ndarray
            Undistorted points array of same shape that input `image_coordinates`.
        """
        logger.debug("[Start] undistort_points")
        
        # Check if the camera has been calibrated
        if not self.is_calibrated():
            raise RuntimeError("Camera has not been calibrated. Cannot undistort points.")

        # Convert pixel coordinates to a proper shape for OpenCV
        reshaped_coordinates = image_coordinates.reshape(-1, 1, 2).astype(numpy.float32)  # Shape (N, 1, 2)

        # Normalized camera coordinates (unit-less values in the camera-centric system)
        # Each point is represented as: (x', y'). The origin is at the optical center and
        # coordinates ate in metric space (not pixels)
        undistorted_norm = cv2.undistortPoints( # Shape: (N, 1, 2)
            src=reshaped_coordinates,
            cameraMatrix=self._camera_mtx,
            distCoeffs=self._dist_coeffs
        )

        # Convert normalized coordinates back to pixel coordinates
        # Takes 2d points from inhomogeneous (x', y') form and converts them to homogeneous (x, y, 1) form
        # Applying [:, 0, :2], only the (x', y') values are extracted, discarding the redundant third coordinate (w=1)
        undistorted_coordinates = cv2.convertPointsToHomogeneous(undistorted_norm)[:, 0, :2]

        # `camera_matrix[:2, :2]` extracts the slice of matrix that contains [ [fx, 0], [0, fx] ]
        # `camera_matrix[:2, 2]` extracts the slice of matrix that contains [ [cx], [cy] ]
        undistorted_coordinates = (undistorted_coordinates @ self._camera_mtx[:2, :2].T) + self._camera_mtx[:2, 2]

        logger.debug("[Finish] undistort_points")
        return undistorted_coordinates

    def rectify_points(self, image_coordinates: NDArray) -> NDArray:
        """
        Transforms pixel coordinates (image space) into real-world XY coordinates
        based on homography matrix.

        Parameters
        ----------
        image_coordinates : numpy.ndarray
            Nx2 array of pixel coordinates (column, row) to be transformed.

        Returns
        -------
        world_coordinates : numpy.ndarray
            Nx2 array of transformed world coordinates (X, Y).
        """
        logger.debug("[Start] rectify_points")

        # Check if the homography matrix is set
        if self._homography_mtx.size == 0:
            raise RuntimeError("Homography matrix is not set. Cannot rectify points.")

        # Check the homography is a 3x3 array
        if self._homography_mtx.ndim != 2 or self._homography_mtx.shape != (3, 3):
            raise ValueError(f"The homography matrix must be a 3x3 array. Got: {self._homography_mtx.tolist()}")

        # Convert pixel points to the required shape (1, N, 2)
        pixel_points = numpy.array([image_coordinates], dtype=numpy.float32)  # Shape (1, N, 2)

        # Apply homography transformation and get the result from shape (1, N, 2) to (N, 2)
        transformed_points = cv2.perspectiveTransform(
            src=pixel_points,
            m=self._homography_mtx
        )[0]  # Extract (N, 2)

        logger.debug("[Finish] rectify_points")
        return transformed_points

    def undistort_image(
            self,
            image: NDArray,
            refine_matrix: bool = False,
            alpha: int = 1,
            crop_roi: bool = False
    ) -> NDArray:
        """
        Undistort image based on intrinsic and extrinsic properties of a camera.

        Parameters
        ----------
        image : NDArray
            Input image array to undistort. Values must be of type `uint8` and can be a 2D array of shape (H, W) or a 3D
            array of shape (H, W, 3). Where H and W are the image height and width respectively.

        refine_matrix : bool, optional
            If True, refine the camera matrix based on `alpha` scaling parameter using 
            `cv2.getOptimalNewCameraMatrix()`. Default is False.

        alpha : int, optional
            Ignored if `refine_camera_matrix=False`. Free scaling parameter and should be between 0 and 1. `alpha=0` means
            that the undistorted image is zoomed and shifted so that only valid pixels are visible (no black areas after
            calibration). `alpha=1` means that the undistorted image is decimated and shifted so that all the pixels from
            the original images from the cameras are retained in the undistorted image (no source image pixels are lost).
            Any intermediate value yields an intermediate result between those two extreme cases. Default is 1.

        crop_roi : bool, optional
            Ignored if `refine_matrix=False`. Region of interest (ROI) inside the undistorted image where all the
            pixels are valid. If `alpha=0` , the ROI cover the whole image. Otherwise, they are likely to be smaller.

        Returns
        -------
        NDArray
            Undistorted image.
        """
        logger.debug("[Start] undistort_image")

        # Check if the camera has been calibrated
        if not self.is_calibrated():
            raise RuntimeError("Camera has not been calibrated.")

        # Image size
        height, width = image.shape[:2]

        # Refine the camera matrix
        new_camera_matrix = None
        valid_roi = (0, 0, width, height)  # Default ROI is the whole image
        if refine_matrix:
            # Verifica el valor de escalado
            if not (0 <= alpha <= 1):
                raise ValueError("El valor de escalado `alpha` debe ser real entre 0 y 1.")

            # Refina la matrix de la camara
            new_camera_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
                cameraMatrix=self._camera_mtx,
                distCoeffs=self._dist_coeffs,
                imageSize=(width, height),
                alpha=alpha,
                newImgSize=(width, height)
            )

        # Calibrate the lens distortion
        image_undistorted = cv2.undistort(
            src=image,
            cameraMatrix=self._camera_mtx,
            distCoeffs=self._dist_coeffs,
            dst=None,
            newCameraMatrix=new_camera_matrix
        )

        # Crop the image to get only valid pixels
        if refine_matrix and crop_roi:
            x, y, w, h = valid_roi
            image_undistorted = image_undistorted[y:y + h, x:x + w]
        
        logger.debug("[Finish] undistort_image")
        return image_undistorted

    def _homography_perspective_transform(self, image: NDArray) -> tuple:
        """
        Applies a perspective transformation to an image using a given homography matrix.

        Parameters
        ----------
        image : numpy.ndarray
            Input image array.

        Returns
        -------
        Tuple[numpy.ndarray]
            - 2D array of transformed x-coordinates (columns).
            - 2D array of transformed y-coordinates (rows).
        """
        logger.debug("[Start] homography_perspective_transform")

        # Generate a grid of (columns, rows) pixel coordinates covering the entire image
        # get_pixel_coordinates(image)
        pixel_columns, pixel_rows = numpy.meshgrid(range(image.shape[1]), range(image.shape[0]))

        # Flatten and stack into Nx2 array of pixel coordinates (column, row) -> (x, y)
        pixel_coordinates = numpy.vstack((pixel_columns.flatten(), pixel_rows.flatten())).T    # Shape (N, 2)

        # Convert to the required shape (1, N, 2)
        # pixel_coordinates = numpy.asarray([pixel_coordinates]).astype(numpy.float32)

        # Apply the perspective transformation using the homography matrix
        transformed_coordinates = self.rectify_points(image_coordinates=pixel_coordinates)

        # Extract the result from shape (1, N, 2) to (N, 2)
        # transformed_coordinates = cv2.perspectiveTransform(
        #     src=pixel_coordinates,
        #     m=homography_matrix
        # )[0]

        # Reshape the transformed coordinates to match the original image grid shape
        transformed_x = transformed_coordinates[:, 0].reshape(pixel_columns.shape)
        transformed_y = transformed_coordinates[:, 1].reshape(pixel_rows.shape)

        # return (
        #     transformed_coordinates[:, 0].reshape(pixel_columns.shape[:2]),
        #     transformed_coordinates[:, 1].reshape(pixel_rows.shape[:2])
        # )
        logger.debug("[Finish] homography_perspective_transform")
        return transformed_x, transformed_y

    def rectify_image(
            self,
            image: NDArray,
            method: str = "nearest",
            output_nodata_value: int = 1,
            mask: Optional[NDArray] = None,
            return_float: bool = False
            # max_grid_size = 1000
    ) -> numpy.ndarray:
        """
        Rectifies an image using a homography and interpolates (using `scipy.interpolate.griddata`) it onto a regular
        grid defined by a bounding box and resolution in a projected coordinate system.

        The output image is clipped to [0, 255] and converted to uint8, ready for saving.

        If no bounding box is provided, the function processes the entire image but limits grid size to avoid memory issues.

        Parameters
        ----------
        image : numpy.ndarray
            Input image array, expected shape (height, width) or (height, width, channels).
            Image must be undistorted if homography matrix was computed based on camera intrinsic/extrinsic properties.

        method : str, optional
            Interpolation method for `scipy.interpolate.griddata` ('nearest', 'linear', 'cubic').
            Defaults to 'nearest'.

        output_nodata_value : int, optional
            Integer value [0-255] to use for pixels outside the valid data area (due to
            interpolation limits or masking). Ignored if `return_float` is True.
            Defaults to 1.

        mask : numpy.ndarray[bool], optional
            2D boolean array with the same dimensions as the output grid (calculated from bbox and resolution).
            If provided and `method` is 'nearest', pixels where mask is False will be set to `output_nodata_value`.
            Ignored otherwise. Defaults to None.

        return_float : bool, optional
            If True, returns a floating point array with NaN values for points outside the convex hull
            of the input points. Otherwise, returns an uint8 array with `output_nodata_value` values for outside points.
            Default is False.

        ?? max_grid_size (int, optional): Maximum number of points in each grid dimension (default: 1000).

        Returns
        -------
        numpy.ndarray
            The rectified image as a NumPy array whose shape is (out_height, out_width) for 2D input image or
            (out_height, out_width, channels) for 3D input image. Data type is float if `return_float` is True and
            contains NaN values for points outside the convex hull of the input points. If `return_float` is False,
            the data type is uint8, containing `output_nodata_value` for outside points.

        Raises
        ------
        ValueError
            If inputs are invalid (e.g., shapes, nodata value range).
        """
        logger.debug("[Start] rectify_image")

        # Check if the homography matrix is set
        if self._homography_mtx.size == 0:
            raise RuntimeError("Homography matrix is not computed.")

        # Input image is expected to be undistorted
        image_undistorted = image

        # Get world coordinates (X, Y) for each pixel (u, v) in the source image
        try:
            transformed_x, transformed_y = self._homography_perspective_transform(image)
        except Exception as e:
            logger.error(f"Error during homography perspective transform: {e}", exc_info=True)
            raise RuntimeError("Homography transformation failed") from e

        # Combine transformed x and y coordinates into a single Nx2 array
        # Each row represents the transformed (x, y) position of a pixel
        transformed_coordinates = numpy.vstack([  # Shape: (columns*rows, 2)
            transformed_x.flatten(),
            transformed_y.flatten()
        ]).T
        logger.debug(f"Transformed X: min={transformed_x.min()}, max={transformed_x.max()}")
        logger.debug(f"Transformed Y: min={transformed_y.min()}, max={transformed_y.max()}")

        # Copy images and transformed coordinate arrays to apply masking
        masked_x = transformed_x.copy()
        masked_y = transformed_y.copy()
        # masked_original_image = image.copy()
        masked_undistorted_image = image_undistorted.copy()

        # Resolution for interpolation grid in meters in X and Y directions
        x_resolution = float(self._xy_resolution[0])
        y_resolution = float(self._xy_resolution[1])

        # If no bounding box is provided, use the entire image as the region
        # !! Currently there are some memory leak issues. Use bbox instead !!
        if self._bounding_box is None:
            logger.warning("No bounding box provided. Processing the entire image with adaptive grid size")
            error_msg = "Currently not supported. Use a defined bounding box"
            logger.error(error_msg)
            raise ValueError(error_msg)

            # Use all pixel coordinates as valid points
            # valid_points = transformed_coordinates
            # inside_indices = numpy.arange(valid_points.shape[0])  # All points are considered valid

            # Define the grid using the min/max values from the transformed coordinates
            # x_min, x_max = numpy.min(transformed_x), numpy.max(transformed_x)
            # y_min, y_max = numpy.min(transformed_y), numpy.max(transformed_y)

            # Compute adaptive resolution to avoid excessive grid size
            # x_resolution = max((x_max - x_min) / max_grid_size, x_resolution)
            # y_resolution = max((y_max - y_min) / max_grid_size, y_resolution)
            # logger.info(f"New grid resolution: X={x_resolution}, Y={y_resolution}")

        # Process using a bounding box
        else:
            # Compute the bounding box coordinates
            x_min, y_min, bbox_width, bbox_height = self._bounding_box
            x_max = x_min + bbox_width
            y_max = y_min + bbox_height
            logger.debug(f"(Xmin={x_min}, Ymin={y_min}, Width={bbox_width}, Height={bbox_height})")

            # Create a rectangle representing the bounding box for point filtering
            bounding_box_rect = mpatches.Rectangle(
                xy=(x_min, y_min),
                width=bbox_width, 
                height=bbox_height,
                linewidth=2, 
                edgecolor="r", 
                facecolor="none"
            )

            # Get the total number of coordinate points and which ones fall inside the bounding box
            total_points = len(transformed_coordinates)
            inside_mask = bounding_box_rect.contains_points(transformed_coordinates)

            # Get indices of inside and outside points
            inside_indices = numpy.arange(0, total_points, 1)[inside_mask]   # Indices of points inside bbox
            outside_indices = numpy.arange(0, total_points, 1)[~inside_mask] # Indices of points outside bbox

            # Convert 1D indices back to 2D image grid coordinates
            row_indices, col_indices = numpy.unravel_index(outside_indices, transformed_x.shape)

            # Mask the pixels outside the bounding box
            masked_x[row_indices, col_indices] = numpy.ma.masked
            masked_y[row_indices, col_indices] = numpy.ma.masked
            # masked_original_image[row_indices, col_indices, :] = numpy.ma.masked
            masked_undistorted_image[row_indices, col_indices] = numpy.ma.masked

            # Get the valid transformed coordinates inside the bounding box
            valid_points = transformed_coordinates[inside_indices, :]

        # Inform the size of grid to generate
        grid_size_x = int(numpy.ceil((x_max - x_min) / x_resolution))
        grid_size_y = int(numpy.ceil((y_max - y_min) / y_resolution))
        logger.debug(f"x_min={x_min}, x_max={x_max}")
        logger.debug(f"y_min={y_min}, y_max={y_max}")
        logger.debug(f"Grid X Size: {grid_size_x}")
        logger.debug(f"Grid Y Size: {grid_size_y}")

        # Create X, Y coordinates
        x_coords = numpy.arange(x_min, x_max, x_resolution)    # ascending order
        y_coords = numpy.arange(y_min, y_max, y_resolution)    # ascending order
        y_coords_flip = y_coords[::-1]                         # Reverse the array (descending order)

        # Define interpolation grid (pixel grid in world coordinates)
        interpolate_grid = numpy.meshgrid(x_coords, y_coords_flip, indexing="xy")
        # xi = (interpolate_grid[0], interpolate_grid[1])
        # print("Grid type:", type(interpolate_grid))
        # print("Grid length:", len(interpolate_grid))

        # Extract values of valid points from the undistorted image
        if image.ndim == 3:
            # RGB values for 3D images (H, W, C)
            rgb_values = numpy.vstack([
                masked_undistorted_image[:, :, 0].flatten()[inside_indices],  # Red channel
                masked_undistorted_image[:, :, 1].flatten()[inside_indices],  # Green channel
                masked_undistorted_image[:, :, 2].flatten()[inside_indices]   # Blue channel
            ]).T  # Shape (N, 3), where N is the number of valid pixels inside bbox
        else:
            # Single channel values for 2D images (H, W)
            rgb_values = masked_undistorted_image.flatten()[inside_indices].reshape(-1, 1) # Shape (N, 1)

        logger.debug("Interpolation started...")
        interpolated_float = scipy.interpolate.griddata(
            points=valid_points,  # Known valid pixel coordinates
            values=rgb_values,    # Corresponding RGB values
            xi=tuple(interpolate_grid),  # Interpolation grid (new coordinates)
            method=method,       # Interpolation method for smooth results (nearest, linear, cubic)
            fill_value=numpy.nan,
            # rescale=False
        )#.clip(0, 255)
        logger.debug("Interpolation finished")

        # Squeeze singleton dimension for grayscale if needed
        if image.ndim == 2:
            interpolated_float = interpolated_float.squeeze(axis=-1)

        # --- Post-processing (NaN -> nodata, mask, clip, type conversion) ---
        if return_float:
            rectified_array = interpolated_float
        else:
            # Replace NaNs resulting from interpolation
            nodata_float = float(output_nodata_value)  # Use float for comparison
            final_image_float = numpy.where(numpy.isnan(interpolated_float), nodata_float, interpolated_float)
            nan_count = numpy.isnan(interpolated_float).sum()
            if nan_count > 0:
                logger.debug(f"Replaced {nan_count} NaN values with nodata value: {output_nodata_value}")

            # Apply optional mask (only for 'nearest' method as other methods blend values)
            if (method == "nearest") and (mask is not None):
                logger.debug("Applying user-provided mask")
                if mask.shape != interpolated_float.shape[:2]:
                    error_msg = (
                        f"Provided mask shape {mask.shape} does not match output image shape {interpolated_float.shape}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Apply mask - where mask is False, set to nodata
                final_image_float[~mask] = nodata_float

            # Clip values to the valid 8-bit range [0, 255]
            final_image_clipped = final_image_float.clip(0, 255)

            # Check if clipping occurred significantly (optional)
            clip_diff = numpy.abs(final_image_float - final_image_clipped).sum()
            if clip_diff > 1e-6 * final_image_clipped.size : # Tolerate tiny float differences
                logger.warning(
                    f"Image values were clipped to [0, 255]. Min/Max before clip: "
                    f"{final_image_float.min():.2f}/{final_image_float.max():.2f}"
                )

            # Convert to final uint8 data type
            rectified_array = final_image_clipped.astype(numpy.uint8)

        logger.debug("[Finish] rectify_image")
        return rectified_array #, interpolate_grid

    def compute_rectification_mask(self, image: NDArray) -> NDArray:
        """
        Creates a mask for valid rectified pixels. 

        Workflow:
            1. Undistort the input image using the camera's intrinsic parameters.
            2. Rectify the undistorted image using the homography matrix and specified bounding box and resolution.
            3. Identify pixels in the rectified image that are valid (i.e., not NaN) and create a boolean mask.

        The output is a boolean mask that identifies valid rectified pixels. A pixel is considered
        valid if it is inside the convex hull of the input points (i.e., pixels whose values are not
        NaN in the rectified result).

        Parameters
        ----------
        image : numpy.ndarray
            The input image to be processed. It is expected to be in a format supported
            for the undistortion and rectification processes.

        Returns
        -------
        NDArray
            A 2D boolean mask indicating valid rectified pixels. Elements of the mask
            are True for valid pixels and False for invalid ones.
        """
        # Undistort and rectify image
        rectified_linear = self.rectify_image(
            image=self.undistort_image(image),
            method="linear",
            return_float=True
        )

        # Invalid mask for grayscale image
        invalid_mask = numpy.isnan(rectified_linear)

        # Color image (RGB)
        if rectified_linear.ndim != 2:
            invalid_mask = invalid_mask.all(axis=2)

        return numpy.logical_not(invalid_mask)
