import logging
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


logger = logging.getLogger("MonitoringSystem")


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
            device: ProcessorDevice,
            cameras: Optional[Dict[str, Camera]] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        self.system_id = id
        self.name = name
        self.location = location
        self.device = device

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

    def start_monitoring(self):
        logger.debug("[Start] start_monitoring")

        # Log the processor device health status
        status_txt: str = self.device.get_health(as_txt=True)
        logger.info(f"Pre-monitoring health status: {status_txt}")

        # Setup all cameras and check if the initialization was successful
        # setup_success = _setup_all_cameras(monitoring_data)
        for camera in self.cameras.values():
            if not camera.setup_device():
                logger.error(f"Failed to setup camera '{camera.camera_id}'")


        # TODO: Continue migrating the below code to use the MonitoringSystem class and its cameras
        # Reintenta en caso de no haber sido exitosa la inicialización
        retry_count = 0
        while (not setup_success) and (retry_count < settings.MONITORING_RETRY_MAX):
            # Espera para reintentar y agrega el registro
            retry_count += 1
            time.sleep(settings.MONITORING_RETRY_TIME)
            logger.warning(f"({retry_count} / {settings.MONITORING_RETRY_MAX}) Reintentando configuracion de camaras")

            # Reintenta inicializar y actualizar los objetos <VideoCapture> para cada camara
            setup_success = _setup_all_cameras(monitoring_data)

        # Finaliza el proceso si no fue exitosa la inicialización
        if not setup_success:
            logger.error("No se pudieron configurar todas las camaras. Fin de la tanda")
            return

        # Obtiene el nombre para el proceso de monitoreo con base en la serie y fecha-hora actual
        recording_name = _get_recording_name(config_data)
        logger.info(f"Tanda de monitoreo: {recording_name}")

        # Configuración del monitoreo concurrente o paralelo
        tasks = []
        multi_cameras = len(monitoring_data) > 1
        for camera_name, camera_data in monitoring_data.items():
            # Nombre de la sub carpeta donde se guardarán los archivos
            subdir_name = recording_name if camera_data["process_kwargs"]["process"] == "captures" else "video"

            # Carpeta de salida para la camara
            camera_dir = os.path.join(
                # .../data/monitoring/captures/camara_izquierda | .../data/monitoring/results/camara_izquierda
                camera_data["output_directory"],
                # 20240611T183000 | "video"
                subdir_name
            )
            camera_dir = str(os.path.normpath(camera_dir))
            os.makedirs(camera_dir, exist_ok=True)

            # Actualiza la ruta de archivo para guardar las imágenes o video
            extension = camera_data["output_extension"]
            filename = f"{recording_name}.{extension}"
            filepath = os.path.join(camera_dir, filename)
            camera_data["process_kwargs"]["output_fpath"] = filepath
            # print("[DEBUG]", camera_data["process_kwargs"])

            # Crea la tarea y la inicia
            # Concurrente: threading.Thread(target, kwargs)
            # Paralelo: multiprocessing.Process(target, kwargs)
            # self._target(*self._args, **self._kwargs)
            if multi_cameras:
                task = threading.Thread(target=record_and_save, kwargs=camera_data["process_kwargs"])
                tasks.append(task)
            else:
                record_and_save(**camera_data["process_kwargs"])

        # Inicia las tareas concurrentes o paralelas y espera a que terminen
        if multi_cameras:
            logger.info("Inicio de multi-tareas: " + get_raspi_status())

            # Inicia todas las tareas
            for task in tasks:
                task.start()

            # Espera a que todas las tareas terminen
            for task in tasks:
                task.join()

            logger.info("Fin de multi-tareas: " + get_raspi_status())

        # Cierra el stream de video (<VideoCapture>) de todas las cámaras
        _release_video_capture_objects(monitoring_data)

        # Muestra recursos de la RasPi después de cerrar los streams
        logger.info("Fin del monitoreo: " + get_raspi_status())
        logger.debug("[Finish] start_monitoring")

        logger.debug("[Finish] start_monitoring")