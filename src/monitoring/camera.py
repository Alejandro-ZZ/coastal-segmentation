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


class Camera:
    """
    Base class for hardware-specific sources of camera snapshots and recordings.
    
    Parameters
    ----------
    camera_id, system_id, name : str
        Stable identifiers and human-readable name of the camera.
    
    image_size : ImageSize
        Pixel dimensions of the camera's output images.
    
    geometry : CameraGeometry
        Geometric properties of the camera, including intrinsic and extrinsic parameters.
    
    device : CaptureDevice
        Hardware-specific capture device for acquiring images or video from the camera.
    
    info : dict, optional
        Additional metadata about the camera, such as description, manufacturer, model, serial number, etc.
    """
    def __init__(
            self, 
            camera_id: str,
            system_id: str,
            name: str,
            image_size: ImageSize,
            geometry: CameraGeometry, 
            device: CaptureDevice,
            info: Optional[dict] = None, 
    ):
        # Profile information
        self.camera_id = camera_id
        self.system_id = system_id
        self.name = name
        self.image_size = image_size
        self.info: Dict[str, Any] = info if info is not None else {}

        # Geometry and capturing device services
        self.geometry = geometry
        self.device = device

        # State information
        self.is_active: bool = False
        self.installed_at: Optional[datetime] = None

    def install(self, date: Optional[datetime] = None):
        """Mark the camera as installed and active."""
        if date is None:
            date = datetime.now()
        
        self.is_active = True
        self.installed_at = date
        logger.info(f"Camera {self.name} ({self.camera_id}) installed at {self.installed_at}")

    def uninstall(self):
        """Mark the camera as uninstalled and inactive."""
        self.is_active = False
        logger.info(f"Camera {self.name} ({self.camera_id}) uninstalled")