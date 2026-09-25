import abc
import logging

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
    Optional
)

from src.monitoring.camera_geometry import CameraGeometry


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
