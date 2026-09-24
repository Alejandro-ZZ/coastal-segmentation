import abc
import cv2
import logging
import numpy

from dataclasses import (
    dataclass, 
    field
)
from datetime import datetime
from pathlib import Path
from numpy.typing import NDArray
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple
)


logger = logging.getLogger("Camera")


@dataclass(frozen=True)
class ImageSize:
    """Pixel dimensions of an image or video frame"""
    width: int
    height: int


@dataclass(frozen=True)
class CameraProfile:
    """Stable identity and non-secret metadata of a physical camera."""
    camera_id: str
    system_id: str
    name: str
    model: str
    native_image_size: ImageSize
    manufacturer: Optional[str] = None
    serial_number: Optional[str] = None
    installed_at: Optional[datetime] = None
    is_active: bool = True
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CameraGeometry:
    """
    Camera geometry configuration:
    
    1. Lens calibration parameters for a camera at one native image size.
    2. Planar image-to-world mapping and output-grid settings for one camera.
    """
    def __init__(self):
        # Lens calibration parameters
        self._camera_mtx: Optional[NDArray] = None
        self._dist_coeffs: Optional[NDArray] = None
        self._calib_meta: Dict[str, Any] = {}

        # Planar image-to-world data
        self._homography_mtx: Optional[NDArray] = None
        self._rectify_mask: Optional[NDArray] = None

    def calibration_params(self) -> Dict[str, Any]:
        """Return the camera calibration parameters and metadata."""
        return {
            "camera_matrix": self._camera_mtx,
            "distortion_coefficients": self._dist_coeffs,
            "calibration_metadata": self._calib_meta
        }

    def calibrate_from_chessboard(
        self,
        filepaths: List[Path],
        pattern_size: Tuple[int, int],
        compute_error: bool = True,
        draw_corners: bool = False,
        refine_corners: bool = True,
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

        draw_corners : bool, optional
            If True, draw found corners and save it in a 'drawn_corners' directory in the same path of the first 
            chessboard file. The output filename is the same of the processed file name with the "_corners" suffix.

        refine_corners : bool, optional
            If True, refines the found corner locations using `cv2.cornerSubPix()`. Default is True.

        Returns
        -------
        Dict[str, Any]
            Dictionary with the camera intrinsic and extrinsic parameters:

            - ``camera_matrix`` (numpy.ndarray): 3x3 floating-point camera intrinsic matrix.
            - ``distortion_coefficients`` (numpy.ndarray): vector coefficients of 4, 5, 8, 12 or 14 elements.
            - ``mean_pixel_error`` (float): mean re-projection pixel error. None If compute_error is False.

            If no corners were found to any chessboard file, return an empty dictionary.
        """
        logger.debug("[Start] calibrate_from_chessboard")
        if len(filepaths) == 0:
            raise ValueError("Empty chessboard pattern files.")
        logger.debug(f"Calibrating with {len(filepaths)} chessboard pattern files")

        # Output calibration parameters
        camera_parameters: Dict[str, Any] = {}

        # Output image files config
        drawn_corners_path: Path = Path()
        if draw_corners:
            # Input file properties
            chessboard_file = filepaths[0]
            pattern_dirname = chessboard_file.parent

            # Output directory for drawn corner images
            drawn_corners_path = pattern_dirname / "drawn_corners"
            drawn_corners_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Drawn corners saved at: '{drawn_corners_path.as_posix()}'")

        # Chessboard pattern size
        points_per_row, points_per_colum = pattern_size

        # Prepare object points, like: (0,0,0), (1,0,0), (2,0,0) ...,(6,5,0)
        objp = numpy.zeros((points_per_row * points_per_colum, 3), numpy.float32)
        objp[:, :2] = numpy.mgrid[0:points_per_colum, 0:points_per_row].T.reshape(-1, 2)

        # Lists to store object points and image points from all the images
        obj_points: List[numpy.ndarray] = []  # 3d point in real world space
        img_points: List[numpy.ndarray] = []  # 2d points in image plane

        # Get the chessboard image shape from the first file
        image_shape = cv2.imread(filepaths[0].as_posix()).shape[:2]

        # Process each chessboard pattern image
        for chessboard_file in filepaths:
            # Read reference image as BGR and gray
            image: NDArray = cv2.imread(chessboard_file.as_posix())
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Check all chessboard images have the same shape
            if image_shape != image_gray.shape:
                logger.warning(
                    f"Chessboard image with different shape for '{chessboard_file.name}'. "
                    f"Expected: {image_shape}. Got: {image_gray.shape} "
                )

            # Finde the chessboard corners
            find_success, corners = cv2.findChessboardCorners(
                image=image_gray,
                patternSize=(points_per_colum, points_per_row),
                corners=None
            )

            # If any corner, save it
            if find_success:
                # Refine detection if required
                if refine_corners:
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

                # Add the object and image points
                obj_points.append(objp)
                img_points.append(corners)

                # If required, draw the found corners
                if draw_corners:
                    # Draw detected corners and save it
                    cv2.drawChessboardCorners(
                        image=image,
                        patternSize=(points_per_colum, points_per_row),
                        corners=corners,
                        patternWasFound=find_success
                    )
                    output_filepath = drawn_corners_path / f"{chessboard_file.stem}_corners{chessboard_file.suffix}"
                    cv2.imwrite(output_filepath.as_posix(), image)
            else:
                logger.warning(f"No corners found for file: '{chessboard_file.name}'")

        # Both lists must have the same number of elements
        assert len(obj_points) == len(img_points), "`obj_points` and `img_points` have different number of elements"

        # No corners found for all chessboard files. Output is an empty dict
        if len(obj_points) == 0 and len(img_points) == 0:
            logger.error("No corners were found for any input chessboard image file")

        # If any corner were detected, process them
        else:
            find_success, camera_matrix, dist_coeffs, rotation_vectors, translation_vectors = cv2.calibrateCamera(
                objectPoints=obj_points,
                imagePoints=img_points,
                imageSize=image_shape[::-1],
                cameraMatrix=None,
                distCoeffs=None
            )

            # Compute the re-projection error
            if compute_error:
                mean_error = 0
                for idx in range(len(obj_points)):
                    # Re-project 3d points in real world space to 2d points in the image plane
                    img_points2, _ = cv2.projectPoints(
                        objectPoints=obj_points[idx],
                        rvec=rotation_vectors[idx],
                        tvec=translation_vectors[idx],
                        cameraMatrix=camera_matrix,
                        distCoeffs=dist_coeffs
                    )
                    error = cv2.norm(src1=img_points[idx], src2=img_points2, normType=cv2.NORM_L2) / len(img_points2)
                    mean_error += error
                mean_error = round(mean_error / len(obj_points), 3)
            else:
                mean_error = None

            # Output camera parameters
            camera_parameters = {
                "camera_matrix": camera_matrix,
                "distortion_coefficients": dist_coeffs,
                "mean_pixel_error": mean_error
            }

        # Muestra cuantas imágenes fallaron al encontrar las esquinas
        # Display a summary of fail to detect corners
        if len(obj_points) != len(filepaths) and len(img_points) != len(filepaths):
            logger.warning(
                f"Corners not detected in all images. Correctly detected: {len(img_points)}. "
                f"Total files: {len(filepaths)}"
            )

        # Populate attributes
        self._camera_mtx = camera_parameters.get("camera_matrix", None)
        self._dist_coeffs = camera_parameters.get("distortion_coefficients", None)
        self._calib_meta = {
            "mean_pixel_error": camera_parameters.get("mean_pixel_error", None),
            "method": "chessboard",
            "pattern_size": pattern_size,
        }

        logger.debug("[Finish] calibrate_from_chessboard")
        return self.calibration_params()

    def compute_homography(
            self,
            image_coordinates: numpy.ndarray,
            world_coordinates: numpy.ndarray,
            camera_matrix: numpy.ndarray,
            distortion_coefficients: numpy.ndarray = numpy.zeros((1, 4)),
            z: float = 0,
            compute_error: bool = False
    ) -> Tuple[numpy.ndarray, Dict[str, float]]:
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

        camera_matrix : numpy.ndarray
            Camera intrinsic matrix. 2D floating-point array of shape (3, 3).

        distortion_coefficients : numpy.ndarray
            Camera distortion coefficients. 2D floating-point array of shape (1, N)
            where 'N' can be 4, 5, 8, 12 or 14 elements.

        z : float
            The Z-coordinate (elevation) in the real-world coordinate system of the
            plane to be rectified. This plane (Z_world = z) will be the plane that
            appears fronto-parallel when projecting an image. Defaults to 0.

        compute_error : bool
            If True, compute re-projection errors in pixels.

        Returns
        -------
        tuple
            -   ``numpy.ndarray``: 3x3 homography matrix.

            -   ``Dict[str, float]``: Re-projection errors in pixels ("mean_pixel_error") and
                meters ("mean_meter_error"). None values are returned if `compute_error` is False.
        """
        logger.debug("[Start] compute_homography")

        image_coordinates = numpy.asarray(image_coordinates).astype(numpy.float32)  # column/x, row/y
        world_coordinates = numpy.asarray(world_coordinates).astype(numpy.float32)  # X, Y, Z
        camera_matrix = numpy.asarray(camera_matrix).astype(numpy.float32)

        # Estimate the camera pose
        # rotation_vector --> shape: (3, 1)
        # translation_vector --> shape: (3, 1)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=world_coordinates, # 3D points (X, Y, Z)
            imagePoints=image_coordinates,  # 2D projections (column, row)
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients
        )

        # Convert a rotation vector (Rodrigues representation) to a rotation matrix
        # Only the first element (index 0) is used. We do not use the jacobian matrix (index 1)
        rotation_matrix = cv2.Rodrigues(rotation_vector)[0] # shape: (3, 3)

        # Assume height of projection plane
        rotation_matrix[:, 2] = rotation_matrix[:, 2] * z

        # Add the translation vector
        rotation_matrix[:, 2] = rotation_matrix[:, 2] + translation_vector.flatten()

        # Compute homography and normalize it
        homography = numpy.linalg.inv(numpy.dot(camera_matrix, rotation_matrix))
        homography = homography / homography[-1, -1]

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
                    cameraMatrix=camera_matrix,
                    distCoeffs=distortion_coefficients
                )
                tot_error += numpy.sum(numpy.abs(image_coordinates[i] - reprojected_image_points)**2)
                total_points += i
            mean_pixel_error = numpy.sqrt(tot_error / total_points)

            # Compute error in meters
            image_coordinates_undistorted = undistort_points(image_coordinates, camera_matrix, distortion_coefficients)
            reprojected_world_points = rectify_points(image_coordinates_undistorted, homography)
            meter_errors = numpy.linalg.norm(reprojected_world_points - world_coordinates[:, :2], axis=1)
            mean_meter_error = numpy.mean(meter_errors)
            best_error_ids = numpy.argsort(meter_errors)
            logger.debug(f"Best error IDs: {best_error_ids}")

            mean_pixel_error = round(float(mean_pixel_error), 4)
            mean_meter_error = round(float(mean_meter_error), 4)
        else:
            mean_pixel_error = None
            mean_meter_error = None

        # Output errors
        mean_errors = {
            "mean_pixel_error": mean_pixel_error,
            "mean_meter_error": mean_meter_error
        }

        logger.debug("[Finish] compute_homography")
        return homography, mean_errors





class Camera(abc.ABC):
    """Base class for hardware-specific sources of camera snapshots and recordings."""
    def __init__(self, profile: CameraProfile, geometry: CameraGeometry):
        self.profile = profile
        self.geometry = geometry

    @abc.abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """Return the current availability and diagnostics of this device."""
        raise NotImplementedError()

    @abc.abstractmethod
    def connect(self) -> None:
        """Open the hardware connection or video stream."""
        raise NotImplementedError()

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Release all hardware resources and close the connection."""
        raise NotImplementedError()

    @abc.abstractmethod
    def capture_snapshot(self) -> NDArray:
        """Capture one in-memory frame from the physical device."""
        raise NotImplementedError()

    @abc.abstractmethod
    def begin_recording(self) -> bool:
        """Start recording and return if the recording was successfully started."""
        raise NotImplementedError()

    @abc.abstractmethod
    def finish_recording(self) -> Path:
        """Stop a recording session and return the path to the saved video file."""
        raise NotImplementedError()
