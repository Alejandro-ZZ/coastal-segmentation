import abc
import cv2
import logging
import numpy

from pathlib import Path
from numpy.typing import NDArray
from typing import (
    Any,
    Dict,
    Optional,
    Union
)


logger = logging.getLogger("CaptureDevice")


class CaptureDevice(abc.ABC):
    """Base class for hardware-specific sources of camera snapshots and recordings."""
    @abc.abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """Return the current availability and diagnostics of this device."""
        raise NotImplementedError()

    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Open the hardware connection or video stream. Return True if 
        successful, False otherwise.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Release all hardware resources and close the connection."""
        raise NotImplementedError()

    @abc.abstractmethod
    def start_streaming(self):
        """Start streaming video from the physical device displaying it in a window or GUI."""
        raise NotImplementedError()

    @abc.abstractmethod
    def capture_frame(self) -> NDArray:
        """
        Capture one in-memory frame from the physical device. 
        Array must be a RGB 3D numpy array of shape (height, width, channels)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def capture_video(
        self, 
        duration: int, 
        out_file: Union[str, Path], 
        frames: bool = False, 
        max_fails: int = 3
    ) -> Dict[str, Any]:
        """
        Capture a continuous video stream from the physical device

        Parameters
        ----------
        duration : int
            Duration of the video capture in seconds.
        
        out_file : Path
            File path to the output video file.
        
        frames : bool, optional
            If True, save each captured frame as an individual JPG image file in addition to the video.
        
        max_fails : int, optional
            Maximum number of consecutive frame capture failures before aborting the video capture.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary containing capture statistics or diagnostics, such as the number of frames 
            captured, number of failures, and total capture time.
        """
        raise NotImplementedError()


class OpenCvCaptureDevice(CaptureDevice):
    """
    Capture device implementation using OpenCV for video capture.
    
    References
    ----------
    - OpenCV VideoCapture: https://docs.opencv.org/3.4.20/d8/dfe/classcv_1_1VideoCapture.html
    - OpenCV VideoWriter: https://docs.opencv.org/4.13.0/dd/d9e/classcv_1_1VideoWriter.html
    """
    _VIDEO_PROFILES = {
        ".avi": {"DIVX", "XVID", "MJPG"},
        ".mp4": {"mp4v", "avc1"},
    }

    def __init__(self, filename: Union[str, int], fps: int = 30, fourcc: str = "DIVX"):
        """
        Parameters
        ----------
        filename : str |int
            Video file, image file sequence, capturing device id, or a URL for a video stream.
        
        fps : int, optional
            Framerate for video stream. Default is 30.
        
        fourcc : str, optional
            Four-character code for the video codec. Default is "DIVX". List of codes can be 
            obtained at page: https://fourcc.org/codecs.php.
        """
        # Video file, image file sequence, capturing device id, or a URL for a video stream
        self.filename: Union[str, int] = filename
        
        # Initialized as an empty capture object. Will be opened in connect().
        self.cap: cv2.VideoCapture = cv2.VideoCapture()

        # Codec for video writing
        supported_fourcc = {code for codes in self._VIDEO_PROFILES.values() for code in codes}
        if len(fourcc) != 4:
            raise ValueError("FourCC code must be a 4-character string.")
        if fourcc not in supported_fourcc:
            raise ValueError(f"Unsupported FourCC code ({fourcc}). Supported: {sorted(supported_fourcc)}")
        self.fourcc: str = fourcc

        # Framerate for video stream. Default is 30.
        self.fps: int = fps

    def set_filename(self, filename: Union[str, int]):
        """
        Update the filename or device ID for the capture device. The method will attempt to 
        connect to the new capture device to validate the filename or device ID.

        Parameters
        ----------
        filename : str | int
            New video file, image file sequence, capturing device id, or a URL for a video stream.
        """
        # Update the filename or device ID for the capture device
        self.filename = filename

        # Attempt to connect to the new capture device to validate the filename or device ID
        self.disconnect()
        if not self.connect():
            raise RuntimeError(f"Failed to connect to the new capture device filename: '{self.filename}'")
        else:
            self.disconnect()
    
    def get_health(self) -> Dict[str, Any]:
        conection_info = {
            "filename": self.filename,
            "is_open": self.cap.isOpened(),
            "frame_width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "frame_height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "backend": self.cap.get(cv2.CAP_PROP_BACKEND),
        }

        stream_info = {
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "fourcc": self.cap.get(cv2.CAP_PROP_FOURCC),
        }

        device_info = {
            "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
            "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
            "hue": self.cap.get(cv2.CAP_PROP_HUE),
            "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),       
        }
        return {
            "connection": conection_info,
            "stream": stream_info,
            "device": device_info,
        }

    def connect(self) -> bool:
        """Create an OpenCV stream object for frame capturing."""
        logger.debug("[Start] connect")

        # Attempt to open the capture device
        connect_success = False
        try:
            self.cap = cv2.VideoCapture(self.filename)
            if not self.cap.isOpened():
                logger.error(f"Failed to open capture device: {self.filename}")
            else:
                connect_success = True
        except Exception as e:
            logger.error(f"Exception occurred while connecting to capture device: {e}")

        # If connection failed, ensure resources are released
        if not connect_success:
            self.disconnect()
            
        logger.debug("[Finish] connect")
        return connect_success
    
    def disconnect(self):
        """Release the OpenCV stream object and any associated resources."""
        logger.debug("[Start] disconnect")
        self.cap.release()
        logger.debug("[Finish] disconnect")

    def capture_frame(self) -> NDArray:
        logger.debug("[Start] capture_frame")

        # Dummy output array
        read_frame: NDArray = numpy.array([])
        
        # Connect to the capture device if it's not already connected
        if not self.cap.isOpened() and not self.connect():
            raise RuntimeError("Failed to connect to the capture device.")
        
        # Read a frame from the capture device
        read_success, read_frame = self.cap.read()

        # Log reading error
        if not read_success:
            logger.error("Failed to capture frame.")
        else:
            # Convert the frame from BGR to RGB format
            read_frame = cv2.cvtColor(read_frame, cv2.COLOR_BGR2RGB)

        logger.debug("[Finish] capture_frame")
        return read_frame

    def capture_video(
            self, 
            duration: int, 
            out_file: Union[str, Path], 
            frames: bool = False, 
            max_fails: int = 3
    ) -> Dict[str, Any]:
        """
        Capture a continuous video stream from the physical device

        Parameters
        ----------
        duration : int
            Duration of the video capture in seconds.
        
        out_file : Path
            File path to the output video file.
        
        frames : bool, optional
            If True, save each captured frame as an individual JPG image file in addition to the video.
        
        max_fails : int, optional
            Maximum number of consecutive frame capture failures before aborting the video capture.
        
        Returns
        -------
        Dict[str, Any]
            A dictionary containing:

            - "frames": Number of frames successfully captured.
            - "duration": Total time taken for the capture in seconds.
            - "failures": Number of consecutive frame capture failures.
            - "frame_rate": Frame rate used for the capture.
            - "frame_width": Width of the captured frames in pixels.
            - "frame_height": Height of the captured frames in pixels.
            - "output_file": Path to the saved video file.
        """
        logger.debug("[Start] capture_video")
        
        # Connect to the capture device if it's not already connected
        if not self.cap.isOpened() and not self.connect():
            raise RuntimeError("Failed to connect to the capture device.")

        # Validate the output file path
        out_file = Path(out_file)
        file_extension = out_file.suffix.lower()
        supported_fourcc = self._VIDEO_PROFILES.get(file_extension)
        if supported_fourcc is None:
            raise ValueError(f"Unsupported video format ({file_extension}). Expected: {list(self._VIDEO_PROFILES.keys())}")
        if self.fourcc not in supported_fourcc:
            raise ValueError(
                f"FourCC code ({self.fourcc}) is incompatible with '{file_extension}' file. "
                f"Supported: {sorted(supported_fourcc)}"
            )

        # Prepare video writer object
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(
            filename=out_file.as_posix(), 
            fourcc=cv2.VideoWriter.fourcc(*self.fourcc), # cv2.VideoWriter_fourcc(*'XVID')
            fps=self.fps, 
            frameSize=(frame_width, frame_height)
        )

        # Compute the number of frames to capture based on duration and fps
        num_frames = duration * self.fps
        n_digits = len(str(num_frames))
        frame_count = 0

        # Initialize video capture loop with failure handling
        n_fails = 0
        out_file.parent.mkdir(parents=True, exist_ok=True)
        start_tick = cv2.getTickCount()
        logger.info(f"Starting video capture for {duration} seconds at {self.fps} fps ({num_frames} frames)")
        while (frame_count < num_frames) and (n_fails < max_fails):
            # Read frame from the capture device
            # `read()` retuns a decoded frame. Combines both: `grab()` --> `retrieve()`
            read_success, read_frame = self.cap.read()
            
            # Process the read frame
            if not read_success:
                # Increment failure count
                n_fails += 1
            else:
                # Reset failure count on successful read
                n_fails = 0

                # Write the frame to the video file and optionally save it as image
                video_writer.write(read_frame)
                if frames:
                    frame_filename = f"{out_file.stem}_frame{frame_count:0{n_digits}d}.jpg"
                    cv2.imwrite(
                        filename=out_file.with_name(frame_filename).as_posix(), 
                        img=read_frame
                    )

                # Update frame progress
                frame_count += 1

                # Control the capture rate to match the desired fps
                elapsed_time = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
                expected_time = frame_count / self.fps
                while elapsed_time < expected_time:
                    # Free up buffer for the next frame by grabbing it without decoding
                    self.cap.grab()
                    elapsed_time = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()

        # Release the video writer
        captured_time = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
        video_writer.release()

        # Capture summary
        video_info = {
            "frames": frame_count,
            "duration": captured_time,
            "failures": n_fails,
            "frame_rate": self.fps,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "output_file": out_file.as_posix(),
        }
        if (n_fails >= max_fails) or (frame_count < num_frames):
            logger.warning(f"Video capture stopped early. Frames: {frame_count}/{num_frames}. Failures: {n_fails}/{max_fails}")
        
        logger.debug("[Finish] capture_video")
        return video_info

    def start_streaming(self):
        """
        Start streaming video from the capture device and display it in a window.
        The streaming will continue until the user presses the 'q' key to exit.
        """
        logger.debug("[Start] start_streaming")
        if not self.cap.isOpened() and not self.connect():
            raise RuntimeError("Failed to connect to the capture device.")

        # Start streaming video from the capture device and display it in a window
        while True:
            # Read a frame from the capture device
            read_success, read_frame = self.cap.read()
            if not read_success:
                raise RuntimeError("Failed to read frame from capture device.")
            
            # Display the frame in a window
            cv2.imshow('Video Stream', read_frame)
            
            # Check for user input to exit the streaming loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        logger.debug("[Finish] start_streaming")


class RtspCameraDevice(OpenCvCaptureDevice):
    """Capture device implementation for RTSP streams using OpenCV."""
    def __init__(self, rtsp_fmt: str, rtsp_params: Dict[str, Any], fps: int = 30, fourcc: str = "DIVX"):
        """
        Parameters
        ----------
        rtsp_fmt : str
            RTSP URL format string. Use placeholders for username, password, ip_address, and port, e.g.,
            "rtsp://{username}:{password}@{ip_address}:{port}/stream".
        
        rtsp_params : Dict[str, Any]
            Parameters for the RTSP URL format string. 
            Common keys include: "username", "password", "ip_address", and "port" (e.g., 554).
        
        fps : int, optional
            Framerate for video stream. Default is 30.
        
        fourcc : str, optional
            Four-character code for the video codec. Default is "DIVX". List of codes can be 
            obtained at page: https://fourcc.org/codecs.php.
        """
        # Initialize the base class with a placeholder filename. 
        # The actual RTSP URL will be set later.
        super().__init__(filename=rtsp_fmt, fps=fps, fourcc=fourcc)

        # Set connection URL format and parameters for RTSP stream
        self._format = rtsp_fmt
        self._params = rtsp_params

        # Set the RTSP URL and update the filename in the base class
        self._url: str = ""
        self._set_url()
        
    def _set_url(self):
        """Check for expected placeholders in the RTSP format string"""
        for placeholder in self._params.keys():
            if f"{{{placeholder}}}" not in self._format:
                raise ValueError(f"RTSP format string is missing placeholder for '{placeholder}'.")

        # Set the RTSP URL using the provided parameters
        self._url = self._format.format(**self._params)

        # Update the filename in the base class to reflect the new RTSP URL
        self.set_filename(self._url)

    def set_conection(self, format: Optional[str] = None, params: Optional[dict] = None):
        """
        Update the RTSP connection URL.

        Parameters
        ----------
        format : str, optional
            New RTSP URL format string. 
            If None (default), the existing format will be used.
        
        params : dict, optional
            New parameters for the RTSP URL format string. 
            If None (default), the existing parameters will be used.
        """
        # Update the RTSP format and parameters
        if format is not None:
            self._format = format
        if params is not None:
            self._params = params

        # Update the RTSP URL and filename
        self._set_url()