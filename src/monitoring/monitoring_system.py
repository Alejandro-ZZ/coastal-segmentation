import logging
import multiprocessing
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
    Tuple,
    Union
)

from src.monitoring.camera import Camera
from src.monitoring.compute_device import ComputeDevice
from src.utils import retry_process


logger = logging.getLogger("MonitoringSystem")


class SystemConfig():
    """
    Parameters
    ----------
    dt_format : str
        The format string for the date and time used in the monitoring 
        system (e.g., "%Y-%m-%d %H:%M:%S").
    
    reference_dt : str
        The reference date and time for the monitoring system. Helps to 
        detect a potential issue with the system's clock.

    is_enabled : bool
        Whether the monitoring system is enabled or not. If False, 
        the system will not perform any monitoring tasks.
    
    sampling_range : Tuple[str, str]
        The start and end times for the monitoring system's sampling period. 
        This defines the time window during which the system is actively monitoring, 
        avoiding unnecessary resource usage outside of this period.
    """
    def __init__(
            self, 
            dt_format: str, 
            reference_dt: str, 
            sampling_range: Tuple[str, str],
            is_enabled: bool = True, 
            delta_minutes: float = 30.0,
            min_disk_space: float = 0.1,  # Minimum disk space needed (in GB) for a monitoring session
    ):
        # Set the datetime format and validate the datetimes
        self.dt_format: str = dt_format
        self._check_datetimes({
            "reference": reference_dt,
            "minimum sampling range": sampling_range[0],
            "maximum sampling range": sampling_range[1]
        })

        # Set the reference datetime and sampling range
        self.reference_dt: datetime = datetime.strptime(reference_dt, dt_format)
        self.sampling_range: Tuple[datetime, datetime]
        self.set_sampling_range(sampling_range)

        self.delta_minutes: float = delta_minutes
        self.is_enabled: bool = is_enabled
        self.min_disk_space: float = min_disk_space
        self.reboots: List[str] = []

    def _check_datetimes(self, datetimes: Dict[str, str]):
        """Check if the provided datetimes are valid according to the specified datetime format"""
        # Validate the datetime format
        try:
            datetime.now().strftime(self.dt_format)
        except Exception as e:
            raise ValueError(f"Invalid datetime format: '{self.dt_format}'") from e

        # Validate config datetimes
        for dt_name, dt_str in datetimes.items():
            try:
                datetime.strptime(dt_str, self.dt_format)
            except Exception as e:
                raise ValueError(
                    f"Invalid config datetime for '{dt_name}'. "
                    f"Datetime string: {dt_str}. Format: {self.dt_format}"
                ) from e    

    def register_reboot(self):
        """Register a reboot event in the monitoring system's metadata."""
        self.reboots.append(datetime.now().strftime(self.dt_format))

    def set_reference_datetime(self, reference_dt: str):
        """Set the reference datetime for the monitoring system."""
        self._check_datetimes({"reference": reference_dt})
        self.reference_dt = datetime.strptime(reference_dt, self.dt_format)

    def set_sampling_range(self, sampling_range: Tuple[str, str]):
        """Set the sampling range for the monitoring system."""
        self._check_datetimes({
            "minimum sampling range": sampling_range[0],
            "maximum sampling range": sampling_range[1]
        })
        self.sampling_range = (
            datetime.strptime(sampling_range[0], self.dt_format),
            datetime.strptime(sampling_range[1], self.dt_format)
        )
        if self.sampling_range[0] >= self.sampling_range[1]:
            raise ValueError(f"Invalid sampling range: {sampling_range}. The minimum must be less than the maximum.")

    def is_ready(self) -> bool:
        """
        Check if the monitoring system is ready for operation based on its configuration.
        
        Returns
        -------
        bool
            True if the system is ready, False otherwise.
        """
        system_ready = True
        current_time = datetime.now()
    
        # Check if the system is enabled
        if not self.is_enabled:
            logger.warning("Monitoring system is disabled.")
            system_ready = False

        # Check if the current time is within the sampling range
        elif not (self.sampling_range[0] <= current_time <= self.sampling_range[1]):
            logger.warning(f"Current time '{current_time}' is outside the sampling range: {self.sampling_range}.")
            system_ready = False

        return system_ready


# TODO: Implement repository handling for saving recording data, snapshots, and metadata
class MonitoringSystem:
    """
    Aggregate representing a site that owns one or more physical cameras.
    
    Parameters
    ----------
    id, name, location : str
        Stable identifiers and human-readable name of the monitoring system, as well as its physical location.
    
    computer : ComputeDevice    
        The computing device responsible for running the monitoring system and managing the cameras.
    
    config : SystemConfig
        Configuration settings for the monitoring system.

    cameras : dict, optional
        Mapping of camera IDs to Camera objects representing the physical cameras owned by the monitoring system.
    """
    def __init__(
            self, 
            id: str, 
            name: str, 
            location: str,
            computer: ComputeDevice,
            config: SystemConfig,
            cameras: Optional[Dict[str, Camera]] = None,
    ):
        self.system_id = id
        self.name = name
        self.location = location

        self.computer = computer
        self.config = config

        self.is_active = True
        self.cameras: Dict[str, Camera] = cameras if cameras is not None else {}
        
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

    def log_reboot(self):
        """
        Log the system reboot event with a timestamp and relevant metadata.
        
        This method can be called whenever the monitoring system is restarted or rebooted, 
        allowing for tracking of system uptime and potential issues related to reboots.
        """
        # Log in system logs
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(f"Monitoring system '{self.name}' (ID={self.system_id}) rebooted at {timestamp}")
        self.computer.log_health(level="WARNING")
        self.config.register_reboot()

    def capture_snapshots(self, out_path: Optional[Path]) -> Dict[str, NDArray]:
        """
        Capture snapshots from all cameras in the monitoring system and save them to disk if an output path is provided.

        Parameters
        ----------
        out_path : Path, optional
            Directory path where the snapshots will be saved. If None (default), snapshots will not be saved to disk.
            If a path is provided, images will be saved in PNG format as: {datetime}_snapshot_{camera_id}.jpg.
            `datetime` is the current date and time in the format "yyyyMMdd_HHMMSS".
        
        Returns
        -------
        Dict[str, NDArray]
            A dictionary mapping camera IDs to their corresponding captured snapshots as numpy arrays.
        """
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        snapshots: Dict[str, NDArray] = {}

        for camera_id, camera in self.cameras.items():
           read_frame = camera.device.capture_frame()
           
           snapshots[camera_id] = read_frame

           if out_path is not None:
               out_file = out_path / f"{date_str}_snapshot_{camera_id}.png"
               skimage.io.imsave(fname=out_file.as_posix(), arr=read_frame)

        return snapshots


    # TODO: Implement
    def ready_to_monitor(self) -> bool:
        # Available computer disk space
        # ---------------------------------
        if self.computer.available_disk_space(percent=False) > self.config.min_disk_space:
            logger.error("Insufficient disk space on the computing device. Aborting monitoring session.")
            return False


        # Computer device datetime
        # ---------------------------------
        time_delta = self.config.delta_minutes * 60
        time_ref = self.config.reference_dt
        time_ok_fnc = lambda _: abs((self.computer.current_datetime() - time_ref).total_seconds()) < time_delta
        computer_time_ok = time_ok_fnc(None)
        if not computer_time_ok:
            # Retry setting the computer datetime
            computer_time_ok = retry_process(
                process=lambda: self.computer.setup_datetime(),
                success=time_ok_fnc,
                max_retries=3,
                retry_delay=5
            )
        if not computer_time_ok:
            logger.error("Invalid computer datetime. Aborting monitoring session.")
            return False
        else:
            # Update the reference datetime in the monitoring system configuration
            now_datetime = self.computer.current_datetime()
            self.config.set_reference_datetime(now_datetime.strftime(self.config.dt_format))


        # Monitoring system operability
        # ---------------------------------
        if not self.config.is_ready():
            logger.error("Monitoring system is not ready. Aborting monitoring session.")
            return False


        # Cameras operability
        # ---------------------------------
        # TODO: We could let the monitoring process start with at least one camera ready
        # Check if all cameras are ready for monitoring, retrying if necessary
        cameras_ready = retry_process(
            process=lambda: [ camera.is_ready() for camera in self.cameras.values() ],
            success=lambda results: all(results),
            max_retries=3,
            retry_delay=30,
            on_retry=lambda attempt, max_retires: logger.warning(f"Retrying camera setup ({attempt}/{max_retires})"),
        )
        if not cameras_ready:
            logger.error("Could not set up cameras successfully. Aborting monitoring session.")
            return False

        return True

    # TODO: Implement 
    def _get_session_id(config_data: dict) -> str:
        """
        Get the session ID to save outputs. Note that this function also update the "serie" data from the
        config_data["system"].

        Parameters
        ----------
        config_data : dict
        Dictionary with all configuration data for the monitoring system.

        Returns
        -------
        name : str
            Formatted name to save the session ID. Example: 05110004_21_09_14_16_17
        """
        logger.debug("[Start] _get_session_id")

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

        logger.debug("[Finish] _get_session_id")
        return f"{serie_str}_{datetime_str}"  # _{coords_str}"

    # TODO: Implement
    def _record_and_save(self, session_id: str):
        """
        Process of recording and saving data from all cameras in the monitoring system concurrently or in parallel.
        
        Parameters
        ----------
        session_id : str
            The unique identifier of the monitoring session, used to name the output files.
        """        
        # List concurrent or parallel tasks to be executed
        parallel_tasks: List[threading.Thread] = []

        # Create a recording and saving task for each camera in the monitoring system
        for camera_id, camera in self.cameras.items():
            # Prepare output file path for camera data (images or video) 
            out_path = Path(...)
            out_file = out_path / f"{session_id}.{asset_extension}"
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
        self.computer.log_health(level="INFO")
        for parallel_task in parallel_tasks:
            parallel_task.start()

        # Wait for all concurrent or parallel tasks to finish
        for parallel_task in parallel_tasks:
            parallel_task.join()

        # Log the elapsed time and device health status
        logger.info(f"Tasks finished. Elapsed time: {time.perf_counter() - t0:.2f} seconds")
        self.computer.log_health(level="INFO")

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

        # Check if the monitoring system is ready for operation
        if not self.ready_to_monitor():
            logger.warning("Monitoring system is not ready for operation")
        else:
            # Get a session ID based on the current configuration data
            session_id = self._get_session_id()
            logger.info(f"Monitoring session: {session_id}")

            # Start recording and saving data
            self.computer.log_health(level="INFO")
            self._record_and_save(session_id)
            self.computer.log_health(level="INFO")
        
        logger.debug("[Finish] start_monitoring")

    def process_session_assets(self, session_id: str):
        """
        Process the recorded assets from a monitoring session.

        Parameters
        ----------
        session_id : str
            The ID of the monitoring session to process.

        This method is a placeholder for future implementation and may include steps such as:
            - Video processing (e.g., stabilization, enhancement)
            - Image analysis (e.g., object detection, segmentation)
            - Data aggregation and reporting
        """
        logger.debug("[Start] process_session_assets")

        # TODO: Test computer overheat to switch to Threads or to a lower performance mode 
        # List of parallel tasks to be executed
        parallel_tasks: List[multiprocessing.Process] = []

        # Create a processing task for each camera in the monitoring system
        for camera_id, camera in self.cameras.items():
            # Process the recorded assets for only intended cameras (e.g., video cameras)
            if camera_id in self.posprocessing_cameras: # TODO: This attribute is thought to be a Set[str] of camera IDs that are configured for post-processing
                # Internally multiprocessing execute: self._target(*self._args, **self._kwargs)
                task = multiprocessing.Process(
                    target=_process_and_save_stats_from_images,
                    args=(camera_name, camera_data, stats_config)
                )
                parallel_tasks.append(task)

        # Start and wait for all concurrent or parallel tasks to finish
        logger.info(f"Session posprocessing started with {len(parallel_tasks)} tasks")
        self.computer.log_health(level="INFO")
        t0 = time.perf_counter()
        for task in parallel_tasks:
            task.start()
        for task in parallel_tasks:
            task.join()
        logger.info(f"Session posprocessing finished. Elapsed time: {time.perf_counter() - t0:.2f} seconds")
        self.computer.log_health(level="INFO")
        logger.debug("[Finish] process_session_assets")



