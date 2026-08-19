"""Download, install, and preprocess the FRF Duck datasets and ResUNet model."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from typing import Iterable
from typing import Sequence

import numpy
import skimage



ZENODO_ARCHIVES = {
    "train": "https://zenodo.org/records/7075342/files/training_data.zip?download=1",
    "test_c1": "https://zenodo.org/records/7075342/files/test_data_c1.zip?download=1",
    "test_c6": "https://zenodo.org/records/7075342/files/test_data_c6.zip?download=1",
    "model": "https://zenodo.org/records/7075342/files/model.zip?download=1",
}

ARCHIVE_DIRECTORIES = {
    "train": "training_data",
    "test_c1": "test_data_c1",
    "test_c6": "test_data_c6",
    "model": "model",
}

DATASET_NAMES = ("train", "test_c1", "test_c6")

TARGET_SHAPE = (2048, 2448)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for FRF Duck asset preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Download FRF Duck data and ResUNet assets from Zenodo, install them in the "
            "repository layout, convert JPG masks to PNG, and normalize training dimensions."
        )
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing 'assets/' directory (default: current directory).",
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=(*DATASET_NAMES, "model"),
        default=(*DATASET_NAMES, "model"),
        help="Assets to prepare. E.g., train, test_c1, test_c6, model (default: all assets).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Directory for downloaded ZIP files (default: a temporary directory).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use ZIP files already present in --download-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing target directories and regenerated PNG masks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions without downloading, changing files, or preprocessing.",
    )
    return parser


    
# DATASET DOWNLOAD, EXTRACTION, AND INSTALLATION
# -----------------------------------------------
def extract_zip_archive(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP archive while rejecting paths outside its staging directory."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(root):
                raise ValueError(f"Archive contains an unsafe path: {member.filename}")
        archive.extractall(destination)


def _find_extracted_directory(staging_dir: Path, expected_name: str) -> Path:
    """Locate the expected top-level directory after archive extraction."""
    exact_match = staging_dir / expected_name
    if exact_match.is_dir():
        return exact_match

    matching_directories = [path for path in staging_dir.rglob(expected_name) if path.is_dir()]
    if len(matching_directories) == 1:
        return matching_directories[0]
    if not matching_directories:
        raise FileNotFoundError(
            f"Expected extracted directory {expected_name!r} was not found in {staging_dir}."
        )
    raise ValueError(
        f"Archive contains multiple directories named {expected_name!r}: {matching_directories}"
    )


def install_directory(source: Path, destination: Path, overwrite: bool) -> None:
    """Move an extracted dataset into its final project location."""
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination}. Use --overwrite to replace it."
            )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    print(f"Installed: {destination}")



# DATASET PREPROCESSING
# -----------------------------------------------
def clean_jpg_mask(labels_arr, num_classes, filter_size):
    """
    Cleans a noisy JPG segmentation mask using morphological operations.

    Parameters
    ----------
    labels_arr : numpy.ndarray
        2D or 3D numpy array containing class labels. If 3D, only the first channel is used.

    num_classes : int
        Total number of valid classes (default 6, meaning 0 to 5).

    filter_size : int
        Radius of the disk used for morphological opening (refer to "radius" parameter in
        skimage.morphology.disk). Increase if noise is thick; decrease if you are losing thin objects.
    """
    # If the mask was saved as RGB by mistake, grab just the first channel
    if labels_arr.ndim == 3:
        labels_arr = labels_arr[:, :, 0]

    # Store our cleaned binary layers
    cleaned_layers = []

    # One-Hot encode and clean
    footprint = skimage.morphology.disk(filter_size)
    for class_idx in range(num_classes):
        # Create a binary mask for the current class
        binary_mask = (labels_arr == class_idx)
        opened_mask = skimage.morphology.opening(binary_mask, footprint)
        # cleaned_mask = skimage.morphology.closing(opened_mask, footprint) # omit closing to avoid merging nearby classes
        cleaned_mask = opened_mask
        cleaned_layers.append(cleaned_mask)
    
    # Convert list back to a 3D numpy array: shape (num_classes, height, width)
    stacked_layers = numpy.array(cleaned_layers)
    
    # Identify conflicts and gaps
    # Count how many classes claim each pixel.
    # 0 = Unassigned (e.g., was a 6/7, or was erased as noise)
    # >1 = Conflict (classes overlap after morphological operations)
    class_counts = numpy.sum(stacked_layers, axis=0)
    
    # Create a temporary mask assigning the class with the highest index.
    # This gives us a base to work from.
    temp_recombined = numpy.argmax(stacked_layers, axis=0)
    
    # Identify pixels that are "invalid": unassigned (0) or in conflict (>1).
    invalid_pixels = (class_counts != 1)
    print("Total invalid pixels (unassigned or conflict):", numpy.sum(invalid_pixels))
    
    # Spatial Fill (nearest neighbor)
    # We use a distance transform to find the nearest valid pixel for every invalid pixel.
    # The EDT (Exact Euclidean Distance Transform) returns the indices of the nearest
    # '0' (which we define as our valid pixels).
    #
    # `distance_transform_edt` calculates distance to the nearest ZERO.
    # So we pass our invalid pixels as 1 (True) and valid as 0 (False).
    _, nearest_valid_indices = distance_transform_edt(invalid_pixels, return_indices=True)

    # Map the nearest valid indices to our temporary recombined mask to fill the gaps
    final_clean_mask = temp_recombined[tuple(nearest_valid_indices)]
    
    # Force the array to unsigned 8-bit integer (standard for masks)
    final_clean_mask = final_clean_mask.astype(numpy.uint8)

    return final_clean_mask


def convert_jpg2png_masks(labels_dir: Path, overwrite: bool) -> int:
    """Clean FRF Duck JPEG masks and save their class labels as PNG files."""
    # Get all JPEG mask files in the labels directory
    jpg_masks = sorted(
        path 
        for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG") 
        for path in labels_dir.glob(pattern)
    )

    # Convert each JPEG mask to a cleaned PNG mask
    converted = 0
    for jpg_mask_path in jpg_masks:
        # Target PNG mask path
        png_mask_path = jpg_mask_path.with_suffix(".png")

        # Omit conversion if the PNG already exists and overwrite is not requested
        if png_mask_path.exists() and not overwrite:
            print(f"Skipping existing PNG mask: {png_mask_path}")
            continue

        # Read the JPEG mask, clean it, and save as PNG
        labels = skimage.io.imread(jpg_mask_path)
        labels = clean_jpg_mask(labels, num_classes=6, filter_size=1)
        skimage.io.imsave(png_mask_path, labels, check_contrast=False)
        converted += 1
    
    print(f"Converted {converted} JPEG mask(s) to PNG in {labels_dir}")
    return converted


def resize_image(image: numpy.ndarray, output_shape: tuple[int, int]) -> numpy.ndarray:
    """Resize an RGB training image while preserving its uint8 representation."""
    resized = skimage.transform.resize(
        image,
        output_shape=(*output_shape, image.shape[2]),
        preserve_range=True,
        anti_aliasing=True,
    )
    return skimage.img_as_ubyte(resized)


def resize_mask(mask: numpy.ndarray, output_shape: tuple[int, int]) -> numpy.ndarray:
    """Resize a class-label mask with nearest-neighbor interpolation."""
    resized = skimage.transform.resize(
        mask,
        output_shape=output_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return skimage.img_as_ubyte(resized)


def normalize_training_assets(dataset_dir: Path) -> int:
    """Resize noncanonical training images and their masks to TARGET_SHAPE."""
    # Check that the dataset contains the expected directories 
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Training dataset must contain images/ and labels/: {dataset_dir}")

    normalized = 0
    for image_path in sorted(images_dir.glob("*.jpg")):
        # Read and check the training image
        image = skimage.io.imread(image_path)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Training image must be RGB: {image_path}")

        # Check and read the corresponding PNG mask path
        mask_path = labels_dir / f"{image_path.stem}_label.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"No PNG mask found for image {image_path.name}: {mask_path}")
        mask = skimage.io.imread(mask_path, as_gray=True).astype(numpy.uint8)

        # Omit resizing if the image and mask already match the target shape
        if image.shape[:2] == TARGET_SHAPE and mask.shape == TARGET_SHAPE:
            continue

        # Check that the image and mask dimensions match before resizing
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Training image and mask dimensions differ for {image_path.name}: "
                f"{image.shape[:2]} versus {mask.shape}."
            )

        # Resize the training image and mask to the canonical TARGET_SHAPE
        skimage.io.imsave(image_path, resize_image(image, TARGET_SHAPE), check_contrast=False)
        skimage.io.imsave(mask_path, resize_mask(mask, TARGET_SHAPE), check_contrast=False)
        normalized += 1
    
    print(f"Normalized {normalized} training image/mask pair(s) to {TARGET_SHAPE}")
    return normalized


def preprocess_datasets(dataset_dirs: Iterable[Path], overwrite: bool) -> None:
    """Convert all downloaded masks and normalize the training set only."""
    for dataset_dir in dataset_dirs:
        # Dataset must contain a `labels/` directory with JPEG masks to convert to PNG
        labels_dir = dataset_dir / "labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Dataset does not contain labels/: {dataset_dir}")

        # Convert all JPEG masks to cleaned PNG masks
        convert_jpg2png_masks(labels_dir, overwrite)

        # If this is the training dataset, resize all images and masks to the canonical TARGET_SHAPE
        if dataset_dir.name == "train":
            normalize_training_assets(dataset_dir)


def main(argv: Sequence[str] | None = None) -> None:
    """Download and prepare requested FRF Duck assets."""
    # Build the argument parser
    parser = build_parser()
    args = parser.parse_args(argv)
    repository_root = args.repository_root.resolve()
    annotations_root = repository_root / "assets" / "annotations" / "FRF_Duck"
    requested_assets = tuple(dict.fromkeys(args.assets))

    # Validate arguments
    if args.skip_download and args.download_dir is None:
        parser.error("--skip-download requires --download-dir containing the ZIP files.")

    # Handle dry-run mode: print the requested assets and their intended destinations
    if args.dry_run:
        print("[ Dry run ] No files will be downloaded, extracted, or modified. List of requested assets:")
        for asset_name in requested_assets:
            if asset_name == "model":
                destination = repository_root / "results" / "models"
            else:
                destination = annotations_root / asset_name
            print(f"[ Dry run ] \t + {asset_name} --> {destination}")
        if any(asset_name in DATASET_NAMES for asset_name in requested_assets):
            print("[ Dry run ] Additional steps: convert JPEG masks to PNG and normalize only training assets to (2048, 2448)")
        return

    # Main processing: download, extract, install, and preprocess the requested assets
    temporary_directory = None
    if args.download_dir is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="frf_duck_")
        download_dir = Path(temporary_directory.name)
    else:
        download_dir = args.download_dir.resolve()
        download_dir.mkdir(parents=True, exist_ok=True)

    try:
        # List of installed datasets to preprocess after all downloads and extractions are complete
        installed_datasets: list[Path] = []

        # Download, extract, and install each requested asset
        for asset_name in requested_assets:
            # Determine the archive path
            archive_path = download_dir / f"{ARCHIVE_DIRECTORIES[asset_name]}.zip"

            # Download the ZIP archive
            if not args.skip_download:
                source_url = ZENODO_ARCHIVES[asset_name]
                destination = archive_path
                print(f"Downloading: {source_url}")
                with urllib.request.urlopen(source_url) as response, destination.open("wb") as output_file:
                    shutil.copyfileobj(response, output_file)

            # Check that the archive exists (downloaded or pre-existing)
            if not archive_path.is_file():
                raise FileNotFoundError(f"Missing downloaded archive: {archive_path}")

            # Extract the archive to a temporary staging directory and install it in the repository layout
            with tempfile.TemporaryDirectory(prefix=f"frf_duck_{asset_name}_") as extraction_path:
                # Extract the ZIP archive
                staging_dir = Path(extraction_path)
                extract_zip_archive(archive_path, staging_dir)
                source_dir = _find_extracted_directory(staging_dir, ARCHIVE_DIRECTORIES[asset_name])

                # Determine the final destination for the asset
                destination = annotations_root / asset_name
                if asset_name == "model":
                    destination = repository_root / "results" / "models"

                # Install the extracted directory into the repository layout
                install_directory(source_dir, destination, args.overwrite)
                if asset_name in DATASET_NAMES:
                    installed_datasets.append(destination)

        # Preprocess all installed datasets (convert JPEG masks to PNG and normalize training assets)
        preprocess_datasets(installed_datasets, args.overwrite)

    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


if __name__ == "__main__":
    main()