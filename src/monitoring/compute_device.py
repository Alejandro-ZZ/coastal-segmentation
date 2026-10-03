import logging
import pynmea2
import pytz
import serial
import subprocess

from datetime import datetime
from logging import config
from pathlib import Path
from typing import (
    Any,
    Dict,
    Literal,
    Optional,
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

    def current_datetime(self) -> datetime:
        return datetime.now()

    # TODO
    def available_disk_space(self, percent: bool = False) -> float:
        pass

    def serial_port_enabled(self) -> bool:
        """
        Check if the Raspberry Pi serial port is enabled by searching for "enable_uart=1" 
        setup line in the content of the configuration file. 
        
        The options for configuration file paths search are:

        * ``/boot/firmware/config.txt``
        * ``/boot/config.txt``
        """
        logger.debug("[Start] check_serial_port_enabled")

        # Final check output
        is_enabled = False

        # Possible file paths of Raspberry config data to look for
        config_files = {
            "/boot/firmware/config.txt",
            "/boot/config.txt"
        }

        # Try to find the valid config file
        config_file: Optional[Path] = None
        while (config_file is None) and (len(config_files) != 0):
            test_file = Path(config_files.pop())
            if test_file.is_file():
                config_file = test_file
            else:
                logger.warning(f"Config file not found: '{test_file.as_posix()}'")

        # None of tested config files found 
        if config_file is None:
            logger.error(f"None of config file options were found")
        
        # Read config file content 
        else:
            with config_file.open(mode="r", encoding="utf-8") as file:
                # Try to find UART setup line
                config_line = file.readline()
                uart_found = "enable_uart" in config_line
                while (not uart_found) and (config_line != ""):
                    config_line = file.readline()
                    uart_found = "enable_uart" in config_line

                # If config was found, check if UART is enable 
                if uart_found:
                    is_enabled = config_line.strip() == "enable_uart=1"
                else:
                    logger.warning(f"UART setup not found in config file: '{config_file.as_posix()}'")

                # for line in file:
                #     if "enable_uart" in line:
                #         logger.debug("[Finish] check_serial_port_enabled")
                #         return line.strip() == "enable_uart=1"

        logger.debug("[Finish] check_serial_port_enabled")
        return is_enabled

    def get_gps_data(self, port: str) -> dict:
        """
        Get relevant information from GPS serial module, using NMEA sentences 
        from "GPRMC" sentence type. See Notes.

        Parameters
        ----------
        port : str
            Serial device name. Example: "/dev/serial0"

        Returns
        -------
        dict
            Info name and value obtained.

        Notes
        -----
        1.  GPS receiver module gives output in standard National Marine Electronics Association (NMEA) 
            string format. NMEA sentence format starts with "$TTSSS", where "TT" is the talker ID that 
            defines device ("GP" for GPS receivers) and the "SSS" is the talker sentence type. 
            This function uses the Recommended minimum specific GPS data ("RMC") sentence type because 
            it includes the date. 
            
            References:

            - https://www.electronicwings.com/sensors-modules/gps-receiver-module
            - https://en.wikipedia.org/wiki/NMEA_0183#NMEA_sentence_format
            - https://docs.novatel.com/OEM7/Content/Logs/GPRMC.htm

        
        2.  The ``pynmea2`` package parses individual NMEA sentences using the ``parse()`` method and returns 
            an object of type ``NMEASentence`` with attributes like: ``timestamp`` and ``datestamp``.
            
            References: 
            
            - https://github.com/Knio/pynmea2/blob/master/README.md#parsing
            - https://www.electronicwings.com/raspberry-pi/gps-module-interfacing-with-raspberry-pi
        """
        logger.debug("[Start] get_gps_serial_info")

        # Output data
        gps_data: Dict[str, Any] = {}

        # Check serial port config
        if not self.serial_port_enabled():
            logger.error("Serial port is not enabled")

        # Try to read GPS data
        else:
            try:
                # Initialize serial connection (e.g., port="/dev/serial0")
                # https://pyserial.readthedocs.io/en/latest/pyserial_api.html
                gps_serial = serial.Serial(port=port, baudrate=9600, timeout=0.5)

                # Read serial port information (see Note 1 in docstring)
                received_data = gps_serial.readline().decode("utf-8")

                # Check for NMEA format and "GPRMC" type match (see Note 1 in docstring)
                if received_data.startswith("$GPRMC"):
                    # Parse the NMEA formated info (see Note 2 in docstring)
                    nmea_msg = pynmea2.parse(received_data)

                    # Get the GPS data
                    msg_date = nmea_msg.datestamp    # datetime.date | None
                    msg_time = nmea_msg.timestamp    # datetime.time | None
                    # msg_lat =  nmea_msg.latitude     # Grados decimales: float
                    # msg_long = nmea_msg.longitude    # Grados decimales: float

                    # Update output with the GPS data 
                    if msg_date and msg_date:
                        gps_data = {
                            "date": msg_date,
                            "time": msg_time
                        }
                else:
                    logger.warning(f"GPS type info missmatch. Expected: '$GPRMC...'. Got: '{received_data}'")

            except pynmea2.ParseError as e:
                logger.error(f"Could not parse data with `pynmea2`: {e}")
            except Exception as e:
                logger.critical(f"Failed to get GPS data at '{port}' serial port : {e}")
        
        logger.debug("[Finish] get_gps_serial_info")
        return gps_data 

    def setup_datetime(self) -> bool:
        """
        Set the Raspberry Pi system date and time using data from a serial GPS module.
        Assumes that serial port is "/dev/serial0" and returns True if process was successfully.
        """
        logger.debug("[Start] setup_datetime")

        # Process status
        setup_success = False

        # Try to retrieve date and time data
        serial_port = "/dev/serial0"
        gps_data = self.get_gps_data(port=serial_port)
        read_date = gps_data.get("date", None)
        read_time = gps_data.get("time", None)

        # Setup the system date and time
        if (read_date is not None) and (read_time is not None):
            try:
                # Create datatime object in Argentine time zone
                # argentina_dt = utc_dt - timedelta(hours=3)
                # ARGENTINA_TZ = timezone(timedelta(hours=-3))
                utc_dt = datetime.combine(date=read_date, time=read_time)
                argentina_tz = pytz.timezone("America/Argentina/Buenos_Aires")
                argentina_dt = utc_dt.astimezone(argentina_tz)

                # Format datetime as string for linux `date` command compatibility
                formatted_time = argentina_dt.strftime("%Y-%m-%d %H:%M:%S")
                
                # Setup the computer system date and time
                # os.system(f'sudo date --set "{formatted_time}"')
                console_result = subprocess.run(
                    args=["sudo", "date", "--set", formatted_time],
                    check=True, 
                    text=True,
                    capture_output=True 
                )
                setup_success = True
                logger.info(f"System date and time updated to '{formatted_time}'. Stdout: {console_result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.critical(f"Fail to excecute console process. Exit code: {e.returncode}. Stderr: {e.stderr}")
            except Exception as e:
                logger.critical(f"Failed to setup system date and time: {e}")
        else:
            logger.error(f"Date and time data not found in GPS data. Got: {gps_data}")

        logger.debug("[Finish] setup_datetime")
        return setup_success


    