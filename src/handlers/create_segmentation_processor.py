from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import pandas
from sklearn.ensemble import RandomForestClassifier

from src.processors import SegmentationProcessor
from src.utils import load_json_file
from src.utils import save_json_file

# Parameters for reproducibility and parallel processing
RANDOM_STATE = 23
N_JOBS = -1


def _get_PHCO_parameters(n_neighbors: int, colorspace: str) -> dict:
    """Get the parameters for creating a SegmentationProcessor for the PHCO dataset."""
    optimized_params: dict = load_json_file("assets/optimized_params.json").get(str(n_neighbors), {})
    optimized_params.update({"random_state": RANDOM_STATE, "n_jobs": N_JOBS})

    training_file = Path("assets/annotations/PHCO/points_rectify_split.csv")
    training_data = pandas.read_csv(training_file).set_index("Split").loc["train"].copy()

    # Coordinates are float64 as a result of rectification process, but we need them as
    # integers for the SegmentationProcessor
    training_data["Cx"] = training_data["Cx"].astype(int)
    training_data["Cy"] = training_data["Cy"].astype(int)

    classes_config = load_json_file("assets/annotations/PHCO/classes_config.json")

    return dict(
        n_neighbors=n_neighbors,
        colorspace=colorspace,
        classes=classes_config["classes"],
        colors=classes_config["palette"],
        images_path=Path("assets/annotations/PHCO/train_points"),
        annotations_data=training_data,
        output_file=Path(f"results/models/PHCO_{n_neighbors}NN_SegmentationProcessor.pkl"),
        classifier_params=optimized_params
    )

def _get_FRF_Duck_parameters(n_neighbors: int, colorspace: str) -> dict:
    """Get the parameters for creating a SegmentationProcessor for the FRF Duck dataset."""
    optimized_params: dict = load_json_file("assets/optimized_params.json").get(str(n_neighbors), {})
    optimized_params.update({"random_state": RANDOM_STATE, "n_jobs": N_JOBS})

    training_file = Path("results/datasets/FRF_Duck/dataset_points.csv")
    training_data = pandas.read_csv(training_file)

    classes_config = load_json_file("assets/annotations/FRF_Duck/small_classes_config.json")

    return dict(
        n_neighbors=n_neighbors,
        colorspace=colorspace,
        classes=classes_config["classes"],
        colors=classes_config["palette"],
        images_path=Path("assets/annotations/FRF_Duck/train/images"),
        annotations_data=training_data,
        output_file=Path(f"results/models/FRF_Duck_{n_neighbors}NN_SegmentationProcessor.pkl"),
        classifier_params=optimized_params
    )



def create_segmentation_processor(
        n_neighbors: int,
        colorspace: str,
        classes: List[str],
        colors: List[Sequence[int]],
        images_path: Path,
        annotations_data: pandas.DataFrame,
        output_file: Path,
        classifier_params: Optional[Dict[str, Any]] = None
):
    """
    Create and train a SegmentationProcessor for image segmentation tasks.

    Parameters
    ----------
    n_neighbors : int
        The number of neighboring pixels to include in the feature extraction.

    colorspace : str
        The color space to use for feature extraction (e.g., "RGB", "YIQ", "HSV").

    classes : List[str]
        A list of class names corresponding to the annotated classes in the dataset.

    colors : List[Sequence[int]]
        A list of RGB color values corresponding to each class in `classes`. Each color
        should be a sequence of three integers in the range [0, 255].

    images_path : Path
        The directory path containing the training images.

    annotations_data : pandas.DataFrame
        A DataFrame containing the annotations for the training images. Required columns:

            * "ImageFile": The file path of the image (relative to `images_path`).
            * "Cx": The pixel x-coordinate of the annotated point.
            * "Cy": The pixel y-coordinate of the annotated point.
            * "Class": The annotated class name. Must match the names in `classes`.

    output_file : Path
        The file path where the trained SegmentationProcessor model (PKL file) and its metadata
        (JSON file) will be saved.

    classifier_params : Dict[str, Any], optional
        Random Forest classifier parameters to use for training. If None, default parameters
        will be used. See `sklearn.ensemble.RandomForestClassifier` documentation for available
        parameters.
    """
    # processor = SegmentationProcessor(
    #     n_neighbors=n_neighbors,
    #     class_to_color=class_to_color,
    #     classifier_params=classifier_params,
    #     colorspace=colorspace
    # )

    print("Creating and training the SegmentationProcessor with the following parameters:")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  colorspace: {colorspace}")
    print(f"  output_file: {output_file}")

    processor = SegmentationProcessor(
        n_neighbors=n_neighbors,
        classifier=RandomForestClassifier(**(classifier_params or {})),
        classes=classes,
        colors=colors
    )

    # Set preprocessing colorspace
    processor.preprocessing.set_colorspace(colorspace)

    # Train the classifier and evaluate on the training set
    processor.evaluate_classifier(images_path=images_path, annotations_data=annotations_data, do_train=True)

    # Save the processor (as PKL) and its metadata (as JSON)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    processor.to_pkl(output_file.with_suffix(".pkl"))
    save_json_file(output_file.with_suffix(".json"), json_data=processor.get_metadata())


# > python -m src.handlers.create_segmentation_processor
if __name__ == "__main__":
    # Example usage
    # handler_kwargs = _get_PHCO_parameters(n_neighbors=24, colorspace="YIQ")
    handler_kwargs = _get_FRF_Duck_parameters(n_neighbors=24, colorspace="YIQ")

    # Create and save the trained segmentation processor
    create_segmentation_processor(**handler_kwargs)

