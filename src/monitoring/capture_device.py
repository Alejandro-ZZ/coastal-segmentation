import abc

from pathlib import Path
from numpy.typing import NDArray
from typing import (
    Any,
    Dict
)


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
    def capture_frame(self) -> NDArray:
        """Capture one in-memory frame from the physical device."""
        raise NotImplementedError()

    @abc.abstractmethod
    def capture_video(self, duration: int, out_path: Path, frames: bool = False):
        """
        Capture a continuous video stream from the physical device

        Parameters
        ----------
        duration : int
            Duration of the video capture in seconds.
        
        out_path : Path
            Path to the output video file or frames.
        
        frames : bool, optional
            If True, save individual frames as images in the given ``out_path`` directory.
            If False (default), only the video file will be saved.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def start_streaming(self):
        """Start streaming video from the physical device displaying it in a window or GUI."""
        raise NotImplementedError()

