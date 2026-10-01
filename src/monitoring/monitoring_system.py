import skimage.io

from datetime import datetime
from pathlib import Path
from numpy.typing import NDArray
from typing import (
    Any,
    Dict,
    Optional,
    Union
)

from src.monitoring.camera import Camera


class MonitoringSystem:
    """
    Aggregate representing a site that owns one or more physical cameras.
    
    Parameters
    ----------
    id, name, location : str
        Stable identifiers and human-readable name of the monitoring system, as well as its physical location.

    cameras : dict, optional
        Mapping of camera IDs to Camera objects representing the physical cameras owned by the monitoring system.
    
    metadata : dict, optional
        Additional metadata about the monitoring system, such as description, owner, contact information, etc.
    """
    def __init__(
            self, 
            id: str, 
            name: str, 
            location: str,
            cameras: Optional[Dict[str, Camera]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.system_id = id
        self.name = name
        self.location = location

        self.is_active = True
        self.cameras: Dict[str, Camera] = cameras if cameras is not None else {}
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}

    def get_cameras(self) -> Dict[str, Camera]:
        """Return a dictionary of all cameras in the monitoring system."""
        return self.cameras

    def add_camera(self, camera: Camera):
        """Add a Camera object to the monitoring system."""
        camera_id = camera.camera_id
        if camera_id in self.cameras:
            raise ValueError(f"Camera with ID '{camera_id}' already exists in the monitoring system.")
        else:
            self.cameras[camera.camera_id] = camera

    def remove_camera(self, camera_id: str) -> Camera:
        """Remove a Camera object from the monitoring system."""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera with ID '{camera_id}' does not exist in the monitoring system.")
        else:
            return self.cameras.pop(camera_id)


    def capture_snapshots(self, out_path: Optional[Path]) -> Dict[str, NDArray]:
        """
        Capture snapshots from all cameras in the monitoring system and save them to disk if an output path is provided.

        Parameters
        ----------
        out_path : Path, optional
            Directory path where the snapshots will be saved. If None (default), snapshots will not be saved to disk.
            Images will be saved in JPEG format as: {datetime}_snapshot_{camera_id}.jpg.
            `datetime` is the current date and time in the format "dd-mm-yyyy_HH-MM-SS".
        
        Returns
        -------
        Dict[str, NDArray]
            A dictionary mapping camera IDs to their corresponding captured snapshots as numpy arrays.
        """
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        
        snapshots: Dict[str, NDArray] = {}

        for camera_id, camera in self.cameras.items():
           read_frame = camera.device.capture_frame()
           
           snapshots[camera_id] = read_frame

           if out_path is not None:
               out_file = out_path / f"{date_str}_snapshot_{camera_id}.jpg"
               skimage.io.imsave(fname=out_file.as_posix(), arr=read_frame)

        return snapshots
