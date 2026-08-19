import json
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import joblib
import numpy
from sklearn.base import BaseEstimator


logger = logging.getLogger("Utils")


def load_json_file(json_fpath: Union[str, Path]) -> dict:
    logger.debug("[Start] load_json_file")

    # Ensure it's a Path object
    json_fpath = Path(json_fpath)

    # Check that the file exists
    if not json_fpath.exists():
        error_msg = f"JSON file not found. File: '{json_fpath}'"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Read the JSON file
    try:
        with json_fpath.open("rb") as f: # encoding='utf-8'
            json_data = json.load(f)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON. File: '{json_fpath}'. Error: {error}")

    # Check loaded data type
    if not isinstance(json_data, dict):
        raise ValueError(f"Invalid JSON structure, expected a dictionary. File: '{json_fpath}'. Got type: {type(json_data).__name__}")

    logger.debug("[Finish] load_json_file")
    return json_data


def save_json_file(
        out_fpath: Union[str, Path],
        json_data: Union[dict, list],
        indent: Union[int, str] = 4,
        append: bool = False
) -> None:
    """
    Save python dictionary or list of dictionaries to a JSON file.

    Parameters
    ----------
    out_fpath : str | Path
        Filepath to save the JSON file.

    json_data : dict | list[dict]
        Serializable data to save as JSON file.

    indent : int | str, optional
        Integer or string to make JSON array elements and object members pretty-printed with that indent level. An
        indent level of 0, negative, or "" will only insert newlines. If None, selects the most compact representation.
        Using a positive integer indent indents that many spaces per level. If indent is a string (such as "\t"), that
        string is used to indent each level. Defaults to 4.

    append : bool, optional
        If `True` and the `out_fpath` file exists, merges with the existing JSON data instead of overwriting.
        For dicts, updates the existing dict. For lists, extends the existing list. Defaults to False.
    """
    logger.debug("[Start] save_json_file")

    # Ensure it's a Path object
    out_fpath = Path(out_fpath)

    if append:
        if out_fpath.exists():
            logger.debug("Appending data to existing JSON file: {}".format(out_fpath.name))
            existing_data = load_json_file(out_fpath)
            if isinstance(existing_data, list) and isinstance(json_data, list):
                existing_data.extend(json_data)
            elif isinstance(existing_data, dict) and isinstance(json_data, dict):
                existing_data.update(json_data)
            else:
                raise TypeError(
                    f"Cannot append: type mismatch between existing data ({type(existing_data).__name__}) "
                    f"and new data ({type(json_data).__name__})"
                )
            json_data = existing_data
        else:
            logger.warning("Creating new file as `append=True` but `out_fpath` does not exist: '{}'".format(out_fpath))

    with out_fpath.open("w") as f:
        # The "default" param is a function that is called for objects that can’t otherwise be serialized.
        # It should return a JSON encodable version of the object.
        json.dump(json_data, f, default=str, indent=indent)

    logger.debug("[Finish] save_json_file")


def estimator_io(
        filepath: Union[str, Path],
        process: Literal["load", "save"],
        estimator: Optional[BaseEstimator] = None,
) -> Union[BaseEstimator, None]:
    """
    Handle load/save processes for scikit-learn estimator using joblib.

    Parameters
    ----------
    filepath : str | Path
        Estimator filepath to load/save. Filename is expected to be a PKL file.

    process : str
        Process name to handle. Possible options are: ["load", "save"].

    estimator : BaseEstimator, optional
        Scikit-learn estimator to save. Required when `process="save"` and omitted if `process="load"`.
        Defaults to None.

    Raises
    ------
        ValueError: If `estimator=None` when process to handle is "save".

    Returns
    -------
    estimator : BaseEstimator | None
        If process to handle is "load", return a scikit-learn estimator, else return None.
    """
    logger.debug("[Start] estimator_io")

    filepath = Path(filepath)

    # Salida por defecto
    output = None

    # Carga estimador de scikit-learn
    if process == "load":
        output = joblib.load(filepath)
        logger.info(f"Estimador cargado: {filepath.name}")

    # Guarda un estimador de scikit learn
    elif process == "save":
        if estimator is None:
            error_msg = "Debe ingresar un estimador valido a guardar (`estimator=None`)."
            logger.critical(error_msg)
            raise ValueError(error_msg)

        # Guarda el modelo en formato PKL
        joblib.dump(estimator, filepath)

        # Registro de la ubicación guardada
        logger.info(f"Estimador guardado: {filepath}")

    logger.debug("[Finish] estimator_io")
    return output


def get_column_names(n_neighbors: int, color_space: str) -> List[str]:
    """
    Generate a list of names to represent color channels based on the number of neighbors and color space.

    Parameters
    ----------
    n_neighbors : int
        The number of neighboring points to consider. Names are generated from 0 to `n_neighbors` inclusive.

    color_space : str, optional
        A string of three characters representing the color channels (e.g., "rgb" for red, green, and blue channels).
        Default is "rgb".

    Raises
    ------
    ValueError: If the `color_space` parameter is not a string of three characters or contains non-alphabet characters.

    Returns
    -------
    List[str]
        Sting names representing color channels based. Names are formatted as "c_i", where "c" is a character
        from `color_space` (e.g., "r", "g", "b"), and "i" is the neighbor number.

    Example
    -------
        >>> get_column_names(n_neighbors=3, color_space="rgb")
        [
            "r_0", "g_0", "b_0",
            "r_1", "g_1", "b_1",
            "r_2", "g_2", "b_2",
            "r_3", "g_3", "b_3"
        ]
    """
    # Check input parameters
    if len(color_space) != 3:
        raise ValueError(f"Invalid color space `{color_space}`. Must be a three characters string.")
    elif not color_space.isalpha():
        raise ValueError(f"Invalid color space `{color_space}`. Must be an alphabetic string.")

    # Get the color space characters
    # color_space = color_space.lower()
    char1, char2, char3 = [char for char in color_space.upper()]

    # Create the output name list
    column_names = [char1, char2, char3]
    for neighbor_num in range(1, n_neighbors + 1):
        neighbor_str = str(neighbor_num)
        column_names.extend([
            f"{char1}{neighbor_str}",
            f"{char2}{neighbor_str}",
            f"{char3}{neighbor_str}"
        ])

    return column_names


def convert_label2rgb(
        label_array: numpy.ndarray,
        label_to_color: Dict[int, Sequence[int]]
) -> numpy.ndarray:
    """
    Converts a 2D label array of shape (height, width) with values in range [0, n_labels-1] into an RGB color image
    of shape (height, width) with values in range [0, 255].

    Parameters
    ----------
    label_array : numpy.ndarray
        2D array of integer labels.

    label_to_color : Dict[int, Sequence[int]]
        Dictionary mapping integer label values to RGB color sequences (e.g., {0: [255, 0, 0], 1: [0, 255, 0], ...}).

    Returns
    -------
    numpy.ndarray
        3D array representing the RGB color image, dtype=uint8.
    """
    logger.debug("[Start] convert_label2rgb")

    # Unique label array values
    uq_label_values = set(numpy.unique(label_array))

    # Check all array values are in the 'label_to_color' mapping
    missing_labels = uq_label_values - set(label_to_color.keys())
    if len(missing_labels) > 0:
        raise ValueError(f"Missing `label_array` values in the `label_to_color` mapping: {missing_labels}")

    # Create an empty RGB image
    height, width = label_array.shape
    color_image = numpy.empty((height, width, 3), dtype=numpy.uint8)

    # Assign colors to the RGB image based on the label values
    for label_value in uq_label_values:
        color = label_to_color[label_value]
        color_image[label_array == label_value] = color

    logger.debug("[Finish] convert_label2rgb")
    return color_image


