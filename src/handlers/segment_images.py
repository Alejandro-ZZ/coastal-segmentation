import time
from pathlib import Path
from typing import Iterable
from typing import Optional

import joblib
import numpy
import skimage
from skimage.util import img_as_ubyte
from sklearnex import patch_sklearn
patch_sklearn()

from src.processors import SegmentationProcessor


def segment_images(
        image_files: Iterable[Path],
        processor_file: Path,
        output_path: Path,
        refine: Optional[str] = None,
        roi_mask_file: Optional[Path] = None,
        class_priors_file: Optional[Path] = None,
):
    """
    Segment images using a pre-trained SegmentationProcessor.

    Parameters
    ----------
    image_files : Iterable[Path]
        File paths to the input images to be segmented.

    processor_file : Path
        File path to the pre-trained SegmentationProcessor model (PKL file).

    output_path : Path
        Directory where the segmented output images will be saved.

    refine : str, optional
        Refinement method to apply to the segmentation results. Options are:

        - None (default): No refinement.
        - "bayes": Bayesian refinement.
        - "crf": Conditional Random Fields refinement.

    roi_mask_file : Path, optional
        File path to a binary mask image (PNG) that defines the region of interest (ROI)
        for segmentation. If provided, only pixels within the ROI will be segmented.

    class_priors_file : Path, optional
        File path to a NumPy file (NPY) containing class prior probabilities array of shape
        (image_height, image_width, n_classes). Required if `refine` is set to "bayes".
    """
    # Initialize the estimator and optional inputs
    segment_model: SegmentationProcessor = joblib.load(filename=processor_file)
    roi_mask: Optional[numpy.ndarray] = None
    class_priors: Optional[numpy.ndarray] = None

    # Load optional inputs if provided
    if roi_mask_file is not None:
        roi_mask = skimage.io.imread(fname=roi_mask_file, as_gray=True) > 0
    if class_priors_file is not None:
        class_priors = numpy.load(class_priors_file)

    # Segmentation loop
    for idx, image_file in enumerate(image_files, start=1):
        # Image predictions
        t0 = time.perf_counter()
        predict_results = segment_model.predict_image(
            rgb_image=skimage.io.imread(fname=image_file),
            roi_mask=roi_mask,
            class_priors=class_priors,
            refine=refine,
            save_file=output_path / image_file.name
        )
        print(f"  [ {idx:02d} ] {image_file.name} \t time={time.perf_counter() - t0:.2f} seconds")

        # Save color labels as PNG image
        color_labels_path = output_path / "color_labels"
        color_labels_path.mkdir(parents=True, exist_ok=True)
        skimage.io.imsave(
            fname=color_labels_path / f"{image_file.stem}.png",
            arr=img_as_ubyte(predict_results["color_labels"])
        )

# ==========  Handlers for specific datasets  ==========
def PHO_handler():
    """Segment PHCO images"""
    n_neighbors = 24
    data_names = ["train_points", "test_polygons"]
    refine_methods = {
        "RF": None,
        "RF-Bayes": "bayes",
        "RF-CRF": "crf"
    }

    model_file = Path(f"results/models/PHCO_{n_neighbors}NN_SegmentationProcessor.pkl")
    priors_file = Path("assets/annotations/PHCO/class_priors.npy")
    roi_mask_file = Path("assets/annotations/PHCO/roi_rectify_mask.png")

    print("\n==========  PHCO Image Segmentation  ==========")
    print("Number of neighbors:", n_neighbors)
    print(f"Model file: {model_file}")
    print(f"Class priors file: {priors_file}\n")

    for data_name in data_names:
        for refine_name, refine_method in refine_methods.items():
            print(f"Data: {data_name} \t Model: {refine_name} \t Neighbors: {n_neighbors}")
            segment_images(
                image_files=Path(f"assets/annotations/PHCO/{data_name}").glob("*.tif"),
                processor_file=model_file,
                output_path=Path(f"results/predictions/PHCO/{data_name}/{n_neighbors}NN/{refine_name}"),
                refine=refine_method,
                roi_mask_file=roi_mask_file,
                class_priors_file=priors_file if refine_method == "bayes" else None
            )
            print()

def FRF_Duck_handler():
    """Segment FRF Duck images"""
    data_names = ["test_c1", "test_c6"]
    refine_methods = {
        "RF": None,
        "RF-Bayes": "bayes",
        "RF-CRF": "crf"
    }

    model_file = Path("results/models/FRF_Duck_24NN_SegmentationProcessor.pkl")
    priors_file = Path("assets/annotations/FRF_Duck/class_priors.npy")

    print("\n==========  FRF Duck Image Segmentation  ==========")
    print(f"Model file: {model_file}")
    print(f"Class priors file: {priors_file}\n")
    for data_name in data_names:
        for refine_name, refine_method in refine_methods.items():
            # Class priors were computed only for C1 camera (train and test_c1 datasets)
            if refine_method == "bayes" and data_name == "test_c6":
                continue

            print(f"Data: {data_name} \t Model: {refine_name}")
            segment_images(image_files=Path(f"assets/annotations/FRF_Duck/{data_name}/images").glob("*.jpg"),
                           processor_file=model_file,
                           output_path=Path(f"results/segmentations/FRF_Duck/{data_name}/{refine_name}"),
                           refine=refine_method, roi_mask_file=None,
                           class_priors_file=priors_file if refine_method == "bayes" else None)
            print()

# Execution: python -m src.handlers.segment_images
if __name__ == "__main__":
    PHO_handler()
