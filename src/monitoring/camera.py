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
from src.monitoring.capture_device import CaptureDevice


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
    def __init__(self, profile: CameraProfile, geometry: CameraGeometry, device: CaptureDevice):
        self.profile = profile
        self.geometry = geometry
        self.device = device

