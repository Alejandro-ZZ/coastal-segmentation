"""Run a trained SegmentationProcessor on a directory of images."""

from __future__ import annotations

import argparse
import joblib
import numpy
import skimage
import time

from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import Sequence

from src.processors import SegmentationProcessor

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass  # sklearnex is optional; if not installed, continue without patching


# CLI ARGUMENT PARSER
# --------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the segmentation command."""
    parser = argparse.ArgumentParser(
        description="Segment all images in a directory with a trained processor."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern used to select input images (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Trained SegmentationProcessor PKL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for compressed predictions and color-label PNG files.",
    )
    parser.add_argument(
        "--refine",
        choices=("none", "bayes", "crf"),
        default="none",
        help="Optional spatial refinement (default: %(default)s).",
    )
    parser.add_argument(
        "--roi-mask",
        type=Path,
        help="Optional binary image mask; only nonzero pixels are segmented.",
    )
    parser.add_argument(
        "--class-priors",
        type=Path,
        help="Spatial class-prior NPY file; required for --refine bayes.",
    )
    return parser


# MAIN HANDLER DEFINITIONS
# --------------------------------------------------
def _require_file(parser: argparse.ArgumentParser, path: Path, option: str) -> None:
    if not path.is_file():
        parser.error(f"{option} does not exist or is not a file: {path}")

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
            arr=skimage.util.img_as_ubyte(predict_results["color_labels"])
        )


# MAIN EXECUTION
# --------------------------------------------------
def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and segment the selected images."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate input arguments
    if not args.images.is_dir():
        parser.error(f"--images does not exist or is not a directory: {args.images}")
    _require_file(parser, args.model, "--model")
    if args.roi_mask is not None:
        _require_file(parser, args.roi_mask, "--roi-mask")
    if args.class_priors is not None:
        _require_file(parser, args.class_priors, "--class-priors")
    if args.refine == "bayes" and args.class_priors is None:
        parser.error("--class-priors is required when --refine bayes is selected.")

    image_files = sorted(args.images.glob(args.pattern))
    if not image_files:
        parser.error(f"No images match {args.pattern!r} in {args.images}")


    segment_images(
        image_files=image_files,
        processor_file=args.model,
        output_path=args.output,
        refine=None if args.refine == "none" else args.refine,
        roi_mask_file=args.roi_mask,
        class_priors_file=args.class_priors,
    )


if __name__ == "__main__":
    main()