"""Train and serialize a SegmentationProcessor from point annotations."""

from __future__ import annotations

import argparse
import pandas

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

from src.processors import SegmentationProcessor
from src.utils import load_json_file
from src.utils import save_json_file


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for Random Forest processor training."""
    parser = argparse.ArgumentParser(
        description="Train a Random Forest SegmentationProcessor from point annotations."
    )
    parser.add_argument("--images", type=Path, required=True, help="Directory containing training images.")
    parser.add_argument("--annotations", type=Path, required=True, help="CSV with 'ImageFile', 'Cx', 'Cy', and 'Class' columns.")
    parser.add_argument("--classes-config", type=Path, required=True, help="JSON containing classes and/or palette arrays.")
    parser.add_argument("--neighbors", type=int, choices=(0, 8, 24), required=True, help="Neighborhood size.")
    parser.add_argument("--colorspace", default="RGB", help="Preprocessing color space (default: %(default)s).")
    parser.add_argument("--output", type=Path, required=True, help="Output artifacts path: PKL and JSON files.")
    parser.add_argument("--split", help="Optional Split-column value used to select training rows.")
    parser.add_argument(
        "--coerce-coordinates",
        action="store_true",
        help="Convert 'Cx' and 'Cy' values to integers before training.",
    )
    parser.add_argument(
        "--classifier-params",
        type=Path,
        help="JSON object of scikit-learn RandomForestClassifier parameters.",
    )
    return parser


def create_segmentation_processor(
        n_neighbors: int,
        colorspace: str,
        classes: List[str],
        images_path: Path,
        annotations_data: pandas.DataFrame,
        output_file: Path,
        classifier_params: Optional[Dict[str, Any]] = None,
        colors: Optional[List[Sequence[int]]] = None,
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
    
    colors : List[Sequence[int]], optional
            A list of RGB color values corresponding to each class in `classes`. Each color
            should be a sequence of three integers in the range [0, 255].
    """
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




def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and train a SegmentationProcessor."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Check path arguments
    if not args.images.is_dir():
        parser.error(f"--images does not exist or is not a directory: {args.images}")
    if not args.annotations.is_file():
        parser.error(f"--annotations does not exist or is not a file: {args.annotations}")

    # Load and validate the classes configuration
    try:
        classes_config =load_json_file(args.classes_config)
    except Exception as e:
        parser.error(f"Error loading --classes-config option: {e}")
    if "classes" not in classes_config:
        parser.error(f"--classes-config must contain 'classes' key. Got: {classes_config.keys()}")

    # Load and check the annotations CSV
    try:
        annotations = pandas.read_csv(args.annotations)
    except Exception as e:
        parser.error(f"Error loading --annotations option: {e}")
    required_columns = {"ImageFile", "Cx", "Cy", "Class"}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        parser.error(
            "--annotations is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if args.split is not None:
        if "Split" not in annotations:
            parser.error("--split requires a 'Split' column in --annotations.")
        annotations = annotations.loc[annotations["Split"] == args.split].copy()
        if annotations.empty:
            parser.error(f"No annotation rows match --split {args.split!r}.")
    if args.coerce_coordinates:
        annotations["Cx"] = annotations["Cx"].astype("int")
        annotations["Cy"] = annotations["Cy"].astype("int")

    # Load classifier parameters if provided
    classifier_params = None
    if args.classifier_params is not None:
        try:
            classifier_params = load_json_file(args.classifier_params)
        except Exception as e:
            parser.error(f"Error loading --classifier-params option: {e}")


    create_segmentation_processor(
        n_neighbors=args.neighbors,
        colorspace=args.colorspace,
        classes=classes_config["classes"],
        images_path=args.images,
        annotations_data=annotations,
        output_file=args.output,
        classifier_params=classifier_params,
        colors=classes_config.get("palette", None),
    )


if __name__ == "__main__":
    main()