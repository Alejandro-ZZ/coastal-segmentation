import logging
import skimage.io
import time
import threading

from datetime import datetime
from pathlib import Path
from numpy.typing import NDArray
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union
)

from src.monitoring.camera import Camera
from src.utils import retry_process


logger = logging.getLogger("MonitoringSystem")


# TODO: Implement repository handling for saving recording data, snapshots, and metadata
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
            computer: ProcessorDevice,
            cameras: Optional[Dict[str, Camera]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.system_id = id
        self.name = name
        self.location = location
        self.computer = computer

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


    # TODO: Implement 
    def _get_recording_name(config_data: dict) -> str:
        """
        Get the recording sample name to save outputs. Note that this function also update the "serie" data from the
        config_data["system"].

        Parameters
        ----------
        config_data : dict
        Dictionary with all configuration data for the monitoring system.

        Returns
        -------
        name : str
            Formatted name to save the recording name. Example: 05110004_21_09_14_16_17
        """
        logger.debug("[Start] _get_recording_name")

        # Obtiene la fecha y hora actual en el formato definido
        datetime_fmt = config_data["system"]["datetime_fmt"]
        datetime_str = datetime.now().strftime(datetime_fmt)

        # Obtiene el número de serie actual y la cantidad de digits a formatear
        serie = config_data["system"]["serie"]
        n_digits = config_data["system"]["serie_digits"]

        # Numero de serie formateado
        serie_str = str(serie).zfill(n_digits)

        # Actualiza el número de serie en la unidad
        config_data["system"]["serie"] += 1

        # Sobreescribe el archivo JSON con la información actualizada
        save_json_file(settings.MONITORING_CONFIG_FPATH, config_data)
        logger.info(f"Serie de monitoreo actualizada: {config_data['system']['serie']}")

        # Obtiene el formato para las coordenadas geográficas
        # coords_str = format_coordinates(
        #     latitude=config_data["system"]["geo_coords"]["lat"], 
        #     longitude=config_data["system"]["geo_coords"]["long"]
        # )

        logger.debug("[Finish] _get_recording_name")
        return f"{serie_str}_{datetime_str}"  # _{coords_str}"

    # TODO: Implement
    def _record_and_save(self, recording_name: str):
        # Configuración del monitoreo concurrente o paralelo
        
        # List concurrent or parallel tasks to be executed
        parallel_tasks: List[threading.Thread] = []

        # Create a recording and saving task for each camera in the monitoring system
        for camera_id, camera in self.cameras.items():
            # Prepare output file path for camera data (images or video) 
            out_path = Path(...)
            out_file = out_path / f"{recording_name}.{asset_extension}"
            out_path.mkdir(parents=True, exist_ok=True)

            # Connect the camera device
            camera.device.connect()

            # Creates the camera recording thread 
            #   * Concurrent: threading.Thread(target, kwargs)
            #   * Parallel: multiprocessing.Process(target, kwargs)
            # Internally, the thread will call: self._target(*self._args, **self._kwargs)
            parallel_tasks.append(threading.Thread(
                target=camera.device.capture_video, 
                kwargs=dict(duration=, out_file=out_file, frames=False, max_fails=3) # TODO
            ))

        # Start all concurrent or parallel tasks
        t0 = time.perf_counter()
        logger.info(f"Starting {len(parallel_tasks)} concurrent/parallel tasks")
        logger.info(f"Device health status: {self.computer.get_health(as_txt=True)}")
        for parallel_task in parallel_tasks:
            parallel_task.start()

        # Wait for all concurrent or parallel tasks to finish
        for parallel_task in parallel_tasks:
            parallel_task.join()

        # Log the elapsed time and device health status
        logger.info(f"Tasks finished. Elapsed time: {time.perf_counter() - t0:.2f} seconds")
        logger.info(f"Device health status: {self.computer.get_health(as_txt=True)}")

        # Disconnect all camera devices
        for camera in self.cameras.values():
            camera.device.disconnect() 

    # TODO: Implement
    def start_monitoring(self):
        """
        Main monitoring process that orchestrates the following steps:

            1. Check if all cameras are ready for monitoring, retrying if necessary.
            2. Get the recording name based on the current configuration data.
            3. Start recording and saving data from all cameras concurrently or in parallel.
            4. Log the monitoring results and device health status.
        
        Generated media assets will be saved in ?? TODO
        """
        logger.debug("[Start] start_monitoring")
        logger.info(f"Pre-monitoring health status: {self.computer.get_health(as_txt=True)}")

        # TODO: We could let the monitoring process start with at least one camera ready
        # Check if all cameras are ready for monitoring, retrying if necessary
        cameras_ready = retry_process(
            process=lambda: [ camera.is_ready() for camera in self.cameras.values() ],
            success=lambda results: all(results),
            max_retries=3,
            retry_delay=30,
            on_retry=lambda attempt, max_retires: logger.warning(f"Retrying camera setup. Attempt {attempt}/{max_retires}"),
        )

        # Finish the monitoring process if cameras are not ready
        if not cameras_ready:
            logger.error("Could not set up cameras successfully. Aborting monitoring session.")
            return

        # Get the recording name based on the current configuration data
        recording_name = self._get_recording_name()
        logger.info(f"Monitoring session: {recording_name}")

        # Start recording and saving data
        self._record_and_save(recording_name)
        logger.info(f"Monitoring session finished: {self.computer.get_status(as_txt=True)}")
        logger.debug("[Finish] start_monitoring")



