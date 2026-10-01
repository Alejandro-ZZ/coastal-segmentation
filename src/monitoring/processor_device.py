


from typing import (
    Any,
    Dict,
    Union
)


class ProcessorDevice:
    """
    A class representing a processor device for monitoring purposes. Example:
        * Raspberry Pi 4 Model B
        * NVIDIA Jetson Nano
        * Remote server
    """
    def __init__(self):
        pass

    def get_health(self, as_txt: bool = False) -> Union[Dict[str, Any], str]:
        """
        Get the health status of the processor device.

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
        # Compute the CPU usage/temperature, memory RAM usage and disk usage
        pass

    