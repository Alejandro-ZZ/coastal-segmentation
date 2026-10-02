import logging
from typing import (
    Any,
    Dict,
    Literal,
    Union
)


logger = logging.getLogger("ComputeDevice")


# TODO: Implement
class ComputeDevice:
    """
    A class representing a computing device for monitoring purposes. Example:
        * Raspberry Pi 4 Model B
        * NVIDIA Jetson Nano
        * Remote server
    """
    def __init__(self):
        pass

    def get_health(self, as_txt: bool = False) -> Union[Dict[str, Any], str]:
        """
        Get the health status of the computing device.

        Parameters
        ----------
        as_txt : bool, optional
            If True, return the health status as a formatted single line string. 
            Default is False.

        Returns
        -------
        dict | str
            A dictionary containing health metrics if `as_txt` is False, otherwise a formatted string.
        """
        # Compute the CPU usage/temperature, memory RAM usage and disk usage (include warnings for limiting values)
        

        return {}

    def log_health(self, level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"):
        """
        Log the health status of the computing device.

        Parameters
        ----------
        level : str, optional
            The logging level to use. Default is "INFO". Other options include "DEBUG", "WARNING", "ERROR", etc.
        """
        logging_fnc = getattr(logger, level.lower(), logger.info)
        logging_fnc(f"Computer health status: {self.get_health(as_txt=True)}")
    