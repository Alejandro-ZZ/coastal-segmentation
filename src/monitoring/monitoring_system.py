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


# TODO: Every time some configuration is updated, it should be saved in a JSON file to persist the changes across reboots and sessions.
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
    
    sampling_interval : Tuple[str, str]
        The start and end times for the monitoring system's sampling period. 
        This defines the time window during which the system is actively monitoring, 
        avoiding unnecessary resource usage outside of this period.
    """
    def __init__(
            self, 
            dt_format: str, 
            reference_dt: str, 
            sampling_interval: Tuple[str, str],
            is_enabled: bool = True, 
            delta_minutes: float = 30.0,
            min_disk_space: float = 0.1,  # Minimum disk space needed (in GB) for a monitoring session
    ):
        # Set the datetime format and validate the datetimes
        self.dt_format: str = dt_format
        self._check_datetimes({
            "reference": reference_dt,
            "minimum sampling interval": sampling_interval[0],
            "maximum sampling interval": sampling_interval[1]
        })

        # Set the reference datetime and sampling interval
        self.reference_dt: datetime = datetime.strptime(reference_dt, dt_format)
        self.sampling_interval: Tuple[datetime, datetime]
        self.set_sampling_interval(sampling_interval)
        self.sampling_minutes: int = 30 # Default sampling duration in minutes for each monitoring session

        self.delta_minutes: float = delta_minutes   # Maximum allowed difference in minutes between the computer's current datetime and the reference datetime
        self.is_enabled: bool = is_enabled          # Whether the monitoring system is enabled or not
        self.min_disk_space: float = min_disk_space # Minimum disk space needed (in GB) for a monitoring session

        self.session_series: int = 0  # Current session series number
        self.series_digits: int = 8  # Number of digits for the session series number

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

    def set_reference_datetime(self, ref_dt: str):
        """Set the reference datetime for the monitoring system."""
        self._check_datetimes({"reference": ref_dt})
        self.reference_dt = datetime.strptime(ref_dt, self.dt_format)

    def set_sampling_interval(self, interval: Tuple[str, str]):
        """
        Set the sampling interval for the monitoring system.
        
        Parameters
        ----------
        interval : Tuple[str, str]
            A tuple containing the minimum and maximum sampling date and times as strings 
            in the format specified by `dt_format`.
        """
        self._check_datetimes({
            "minimum sampling interval": interval[0],
            "maximum sampling interval": interval[1]
        })
        self.sampling_interval = (
            datetime.strptime(interval[0], self.dt_format),
            datetime.strptime(interval[1], self.dt_format)
        )
        if self.sampling_interval[0] >= self.sampling_interval[1]:
            raise ValueError(f"Invalid sampling interval: {interval}. The minimum must be less than the maximum.")

    def _valid_computer_datetime(self, computer: ComputeDevice) -> bool:
        """
        Check if the computer's current datetime is valid based on the reference 
        datetime and accepted delta in minutes.
        
        Parameters
        ----------
        computer : ComputeDevice
            The computing device whose datetime is to be validated.
        
        Returns
        -------
        bool
            True if the computer's datetime is valid, False otherwise.
        """
        now_dt = computer.current_datetime()
        time_difference = abs((now_dt - self.reference_dt).total_seconds())
        return time_difference < (self.delta_minutes * 60)

    def ready_to_monitor(self, computer: ComputeDevice) -> bool:
        """
        Check if the monitoring system is ready for operation based on its configuration.
        Validations:

        1.  **System enabled:** The monitoring system must be enabled for operation.
        2.  **Computer disk space:** The computer must have sufficient free disk space.
        3.  **Computer datetime:** The computer's datetime must be valid.
        4.  **Sampling interval:** The current time must be within the sampling interval.
        Returns
        -------
        bool
            True if the system is ready, False otherwise.
        """
        # Whether the monitoring system is ready for operation
        system_ready = True
        
        # If current computer datetime is invalid, attempt to set it up automatically (e.g., via internet or GPS module)
        if not self._valid_computer_datetime(computer):
            retry_process(
                process=lambda: computer.setup_datetime(),
                success=lambda results: results,
                max_retries=20,
                retry_delay=10
            )

    
        # System enabled for monitoring
        # ---------------------------------
        if not self.is_enabled:
            logger.warning("Monitoring system is disabled.")
            system_ready = False


        # Computer free disk space
        # ---------------------------------
        elif computer.available_disk_space(percent=False) < self.min_disk_space:
            logger.error(
                f"Insufficient disk space on the computing device. Required: {self.min_disk_space} GB, "
                f"Available: {computer.available_disk_space(percent=False):.2f} GB."
            )
            system_ready = False


        # Computer valid datetime
        # ---------------------------------
        # Check again if setting up the computer's datetime was successful
        elif not self._valid_computer_datetime(computer):
            logger.error("Invalid computer datetime.")
            system_ready = False



        # Valid sampling interval
        # ---------------------------------
        elif not (self.sampling_interval[0] <= computer.current_datetime() <= self.sampling_interval[1]):
            logger.warning(
                f"Current time '{computer.current_datetime()}' is outside "
                f"the sampling interval: {self.sampling_interval}."
            )
            system_ready = False


        # Update the reference datetime in the monitoring system configuration
        # This is done to ensure that the reference datetime is always up-to-date 
        # with the computer's current datetime.
        now_dt = computer.current_datetime()
        self.set_reference_datetime(now_dt.strftime(self.dt_format))

        return system_ready

    def get_session_series(self, update: bool = False, as_txt: bool = False) -> Union[int, str]:
        """
        Get the current session series number, optionally updating it for the next session creation.

        Parameters
        ----------
        update : bool, optional
            If True, increment the session series number. 
            Defaults to False.
        
        as_txt : bool, optional
            If True, return the session series number as a string based on the number of digits specified. 
            Defaults to False.

        Returns
        -------
        int | str
            The current session series number.
        """
        current_series = self.session_series
        
        if update:
            self.session_series += 1
        
        if as_txt:
            current_series = str(current_series).zfill(self.series_digits)
        
        return current_series

    # TODO: Implement
    def create_session(self, computer: ComputeDevice) -> Dict[str, str]:
        """
        1. Create a session ID
        2. Create a session path to save outputs
        3. Update serial for the next session creation
        
        Returns
        -------
        dict
            Dictionary with a new monitoring session:
            
            * ``id``: Formatted name to save the session ID. Example: 05110004_21_09_14_16_17
            * ``path``: Path to save the session outputs. Example: .../data/monitoring/sessions/05110004_21_09_14_16_17
        """
        logger.debug("[Start] create_session")

        datetime_str = computer.current_datetime().strftime(self.dt_format)
        series_str = self.get_session_series(update=True, as_txt=True)
        session_id = f"{series_str}_{datetime_str}"

        # TODO: Base path must be taken from a repository configuration
        session_path = Path(...) / session_id

        logger.debug("[Finish] create_session")
        return {
            "id": session_id,
            "path": session_path.as_posix()
        }


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

    def cameras_ready(self) -> bool:
        """
        Check if all cameras in the monitoring system are ready for operation.
        """
        # TODO: We could let the monitoring process start with at least one camera ready

        # Check if all cameras are ready for monitoring, retrying if necessary
        cameras_ready = retry_process(
            process=lambda: [ camera.is_ready() for camera in self.cameras.values() ],
            success=lambda results: all(results),
            max_retries=3,
            retry_delay=30,
            on_retry=lambda attempt, max_retires: logger.warning(f"Retrying cameras setup ({attempt}/{max_retires})"),
        )
        
        return cameras_ready

    # TODO: Implement
    def _record_and_save(self, session_data: Dict[str, str]):
        """
        Process of recording and saving data from all cameras in the monitoring system concurrently or in parallel.
        
        Parameters
        ----------
        session_data : Dict[str, str]
            A dictionary containing the session data, including the session ID and path.
        """        
        # Get session data
        session_id = session_data["id"]
        session_path = Path(session_data["path"])

        # List concurrent or parallel tasks to be executed
        parallel_tasks: List[threading.Thread] = []

        # Create a recording and saving task for each camera in the monitoring system
        for camera_id, camera in self.cameras.items():
            # Prepare output file path for camera data (images or video) 
            out_path = session_path / f"camera_{camera_id}" ??
            out_file = out_path / f"{session_id}.{asset_extension}"
            out_path.mkdir(parents=True, exist_ok=True)

            # Connect the camera device
            camera.device.connect()

            # Prepare the capture function and its keyword arguments for the camera
            capture_fnc = camera.device.capture_video
            capture_kwargs = dict(
                duration=self.config.sampling_minutes*60, 
                out_file=out_file, 
                frames=False, 
                max_fails=3
            )

            # Include the camera recording thread 
            #   * Concurrent: threading.Thread(target, kwargs)
            #   * Parallel: multiprocessing.Process(target, kwargs)
            # Internally, the thread will call: self._target(*self._args, **self._kwargs)
            parallel_tasks.append( threading.Thread(target=capture_fnc, kwargs=capture_kwargs) )

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

        1. Get the recording name based on the current configuration data.
        2. Start recording and saving data from all cameras concurrently or in parallel.
        3. Log the monitoring results and device health status.

        The monitoring can be aborted in these cases:

        * The monitoring system is not ready for operation (based on system configuration).
        * Not all cameras are ready for monitoring.
        
        Generated media assets will be saved in ?? TODO
        """
        logger.debug("[Start] start_monitoring")

        # Check if the monitoring system is ready for operation
        if not self.config.ready_to_monitor(computer=self.computer):
            logger.warning("Monitoring system is not ready for operation. Aborting monitoring session.")
        
        # Check if all cameras are ready for monitoring
        elif not self.cameras_ready():
            logger.error("Could not set up cameras successfully. Aborting monitoring session.")
        
        # Execute the monitoring process
        else:
            # Get a session ID based on the current configuration data
            session_data = self.config.create_session(computer=self.computer)
            logger.info(f"Monitoring session created: {session_data}")

            # Start recording and saving data
            self.computer.log_health(level="INFO")
            self._record_and_save(session_data)
            self.computer.log_health(level="INFO")
        
        logger.debug("[Finish] start_monitoring")

    # TODO: Implement
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

    # TODO: Implement
    def _process_and_save_stats_from_images(self, camera_name: str, camera_data: dict, stats_config: dict) -> None:
        """
        Process statistics from previously saved images.

        Parameters
        ----------
        camera_name : str
            Name of the camera to be processed.

        camera_data : dict
            Camera data with the following keys:
                * "connection_url": (str) the RTSP url for video stream.

                * "output_directory": (str) directory path to save recording process output.

                * "output_extension": (str) file extension of the recording process output. Example: "jpg"

                * "process_kwargs": (dict) keyword arguments for the recording process function `record_and_save()`.
                    Undefined arguments are set to None because they are supposed to be updated in the monitoring loop.

        stats_config : dict
            Configuration data for the statistics processing. Must have the following keys:
                * "directory" (str): Path to the directory where resulting statistic images are saved.

                * "extension" (str): File extension for the processed statistic images.

                * "names" (list): Statistic names to compute. Valid names are the ones supported by
                    `src.online_stats.OnlineImageStats` class.

                * "fixed_width" (list): Pixel indices to crop images width-wise before processing. Values are as:
                    [min_pixel, max_pixel] and `max_pixel` is non-inclusive.

                * "fixed_height" (list): Pixel indices to crop images height-wise before processing. Values are as:
                    [min_pixel, max_pixel] and `max_pixel` is non-inclusive.
        """
        logger.debug("[Start] _process_and_save_stats_from_images")

        # Ruta y nombre de archivo de la tanda de captura
        # .../base_root/data/monitoring/captures/{camera_name}/20240611T183000/frame-0001_20240611T183000.jpg
        capture_fpath = camera_data["process_kwargs"]["output_fpath"]

        # `capture_fpath` puede ser None si en la ejecución de `start_monitoring` falló la configuración de alguna cámara
        if capture_fpath is not None:
            # Ruta de la carpeta temporal con los archivos capturados
            # .../base_root/data/monitoring/captures/{camera_name}/20240611T183000
            captures_dir = os.path.dirname(capture_fpath)

            # Procesamiento de las estadísticas para cada imagen capturada
            logger.info(f"( {camera_name} ) Procesando estadisticas...")
            start_time = time.time()
            computed_stats = compute_stats_from_images(
                images_dir=captures_dir,
                images_ext=camera_data["output_extension"],
                stats_names=stats_config["names"],
                fixed_width=stats_config["fixed_width"],
                fixed_height=stats_config["fixed_height"],
                use_rgb=False  # Devuelve los resultados en formato BGR
            )
            executing_time = time.time() - start_time
            logger.info(
                f"( {camera_name} ) Tiempo de ejecucion: {int(executing_time // 60)} minutos "
                f"{executing_time % 60:.1f} segundos"
            )

            # Verifica que el procesamiento fue exitoso y guardas las estadísticas como imagen
            if len(computed_stats) > 0:
                logger.info(f"( {camera_name} ) Guardando estadisticas...")
                # Carpeta para guardar los resultados
                output_dir = str(os.path.join(
                    settings.BASE_PATH,
                    os.path.normpath(stats_config["directory"]),
                    camera_name
                ))
                os.makedirs(output_dir, exist_ok=True)

                # Extension para los archivos de salida
                output_ext = stats_config["extension"]

                # Itera cada estadística procesada y la guarda
                for stat_name, stat_arr in computed_stats.items():
                    logger.info(f"( {camera_name} ) Guardando {stat_name}")
                    stat_arr = normalize_array(
                        arr=stat_arr,
                        lower_percentile=1,
                        upper_percentile=99,
                        replace_nan=0.0
                    )
                    os.makedirs(os.path.join(output_dir, stat_name), exist_ok=True)

                    # Guarda la imagen de la estadística
                    capture_name, capture_ext = os.path.splitext(os.path.basename(capture_fpath))
                    stat_image_filename = f"{capture_name}_{stat_name}.{output_ext}"
                    cv2.imwrite(
                        os.path.join(output_dir, stat_name, stat_image_filename),
                        stat_arr.astype(numpy.uint8)
                    )
            else:
                logger.error(f"( {camera_name} ) No se pudieron procesar las imagenes (`computed_stats` vacio)")

            # Borra todos los archivos en la carpeta temporal
            # .../data/monitoring/captures/{camera_name}/20240611T183000
            logger.info(f"( {camera_name} ) Borrando archivos y carpetas...")
            remove_files_in_directory(captures_dir)

            # Borra todas las carpetas vacías (inclusive) de la ruta de captura de la camara
            # .../data/monitoring/captures/camara_izquierda
            remove_empty_folders(camera_data["output_directory"])
        else:
            logger.warning(f"( {camera_name} ) Camara no configurada. Procesamiento de estadisticas omitido")



