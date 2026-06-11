import abc
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional

import numpy
import skimage.io
from matplotlib import pyplot as plt
from tqdm import tqdm


# DATASET GENERATOR PROTOCOLS --------------------------------------------------
class MaskDataset(Iterable[numpy.ndarray], abc.ABC):
    """Abstract base class for a mask dataset (generator protocol)."""
    def __init__(self, filepaths: List[Path]):
        if not filepaths:
            raise ValueError("File paths list cannot be empty.")

        # Initialization -------------
        self.filepaths = filepaths

        # Subclasses must define these properties/methods for validation and loading
        # If not defined, the first loaded mask will set the expected shape for
        # all subsequent masks
        self.expected_shape: Optional[tuple] = None

    def __len__(self):
        return len(self.filepaths)

    @abc.abstractmethod
    def load_mask(self, path: Path) -> numpy.ndarray:
        """Subclasses must define how to read a specific file."""
        raise NotImplementedError("Subclasses must implement 'load_mask' method.")

    @abc.abstractmethod
    def get_classes(self) -> List[str]:
        """Subclasses must define how to provide class names."""
        raise NotImplementedError("Subclasses must implement 'get_classes' method.")

    def get_ignore_label(self) -> Optional[int]:
        """Subclasses can override to specify an ignore label value."""
        return None

    def _validate_mask(self, mask_arr: numpy.ndarray, mask_file: Path):
        """Validate the loaded mask."""
        # Get dataset parameters for validation
        n_classes = len(self.get_classes())
        ignore_label = self.get_ignore_label()

        # Mask is expected to be a 2D array of integer class labels
        if not isinstance(mask_arr, numpy.ndarray):
            raise ValueError(
                f"Expected mask to be a numpy array. Got type: {type(mask_arr)}. File: '{mask_file}'."
            )
        if mask_arr.ndim != 2:
            raise ValueError(f"Expected 2D mask array. Got: {mask_arr.shape}. File: '{mask_file}'.")
        if not numpy.issubdtype(mask_arr.dtype, numpy.integer):
            raise ValueError(
                f"Expected integer mask array. Got dtype: {mask_arr.dtype}. File: '{mask_file}'."
            )

        # All masks must have the same spatial dimensions
        if mask_arr.shape != self.expected_shape:
            raise ValueError(
                f"Unexpected mask dimensions. Expected: {self.expected_shape}. "
                f"Got: {mask_arr.shape}. File: '{mask_file}'."
            )

        # Ignore label value (if specified) must not conflict with valid class labels
        if (ignore_label is not None) and (0 <= ignore_label < n_classes):
            raise ValueError(
                f"Ignore label value ({ignore_label}) conflicts with valid class "
                f"labels [0, {n_classes - 1}]."
            )

        # Mask values must be in the range [0, n_classes - 1] or equal to the ignore label (if specified)
        invalid_mask = (mask_arr < 0) | (mask_arr >= n_classes)
        if ignore_label is not None:
            invalid_mask &= (mask_arr != ignore_label)
        if numpy.any(invalid_mask):
            raise ValueError(
                f"Mask values must be in the range [0, {n_classes - 1}] or equal to the "
                f"ignore label ({ignore_label}). Got unique values: {numpy.unique(mask_arr)}. "
                f"File: '{mask_file}'."
            )

    def __iter__(self) -> Iterator[numpy.ndarray]:
        """Yields masks one by one, keeping memory usage minimal."""
        for filepath in self.filepaths:
            # Load the mask using the subclass-defined method
            mask_arr = self.load_mask(filepath)

            # Assume first loaded mask dimensions as reference
            if self.expected_shape is None:
                self.expected_shape = mask_arr.shape

            # Validate the loaded mask against expected properties
            self._validate_mask(mask_arr, filepath)

            yield mask_arr

class DuckMaskDataset(MaskDataset):
    """Implementation for FRF Duck label files."""
    def __init__(self, filepaths: List[Path], use_train_classes: bool = True):
        super().__init__(filepaths)
        self.use_train_classes = use_train_classes
        self.expected_shape = (2048, 2448)
        self.remap_classes: Dict[int, int] = {
            0: 0,  # background -> background
            1: 1,  # wetsand -> sand
            2: 1,  # drysand -> sand
            3: 2,  # vegetation -> vegetation
            4: 3,  # heavies -> lag
            5: 3  # lag -> lag
        }

    def get_classes(self) -> List[str]:
        """Dataset class names based on dataset"""
        if self.use_train_classes:
            class_names = ["background", "wetsand", "drysand", "vegetation", "heavies", "lag"]
        else:
            class_names = ["background", "sand", "vegetation", "lag"]
        return class_names

    def _preprocess_mask(self, mask_arr: numpy.ndarray) -> numpy.ndarray:
        # This is needed as test classes only include a small subset of the training classes
        if not self.use_train_classes:
            for original_value, new_value in self.remap_classes.items():
                mask_arr[mask_arr == original_value] = new_value

        # There are some images with 2448x2048 resolution and some with 1000x837 resolution.
        # Upsample the smaller ones with nearest neighbor interpolation.
        if mask_arr.shape != self.expected_shape:
            mask_arr = skimage.transform.resize(
                image=mask_arr,
                output_shape=self.expected_shape,
                order=0,                # nearest neighbor interpolation to preserve class labels
                preserve_range=True,    # keep original value range (important for class labels)
                anti_aliasing=False     # no antialiasing for nearest neighbor
            ).astype(numpy.uint8)
        return mask_arr

    def load_mask(self, path: Path) -> numpy.ndarray:
        mask_arr = skimage.io.imread(fname=path, as_gray=True)
        return self._preprocess_mask(mask_arr.astype(numpy.uint8))

class NpzMaskLoader(MaskDataset):
    """
    Concrete implementation for NPZ files.

    Parameters
    ----------
    mask_key : str
        The key in the NPZ file that contains the mask array.
    """
    def __init__(self, filepaths: List[Path], mask_key: str):
        super().__init__(filepaths)
        self.mask_key = mask_key

    def get_classes(self) -> List[str]:
        return []

    def load_mask(self, path: Path) -> numpy.ndarray:
        return numpy.load(path)[self.mask_key]


# HANDLER LOGIC --------------------------------------------------
def save_priors_figure(class_priors: numpy.ndarray, class_names: List[str], output_file: Path):
    max_cols = 3    # maximum number of columns per row
    num_classes = len(class_names)
    n_cols = min(max_cols, num_classes)
    n_rows = int(numpy.ceil(num_classes / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True, sharey=True)

    # Flatten axes for easier indexing, even if 1 row or 1 col
    axs = axs.flatten() if num_classes > 1 else [axs]

    for idx, class_name in enumerate(class_names):
        axs[idx].imshow(class_priors[:, :, idx], cmap="viridis")
        axs[idx].set_title(class_name)
        # axs[idx].grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.6)

    # Hide subplot axes
    for idx in range(len(axs)):
        axs[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved class priors figure to: '{output_file}'")

def compute_class_priors(
        labels_iterable: MaskDataset,
        n_classes: int,
        fill_value: Optional[float] = None
) -> numpy.ndarray:
    """
    Computes class prior probabilities for each pixel based on the provided label files.

    Parameters
    ----------
    labels_iterable : Iterable[numpy.ndarray]
        An iterable that yields 2D numpy arrays of shape (height, width) containing integer
        class labels for each pixel.

    n_classes : int
        The total number of classes. Class labels are expected to be integers in the
        range [0, n_classes - 1].

    fill_value : Optional[float], optional
        Value to fill pixels that have zero count for all classes (e.g., unlabeled pixels).
        If None (default), such pixels will be assigned a uniform distribution over classes.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (height, width, n_classes) with class prior probabilities for each pixel.
    """
    # Lazy initialization of class counts array based on the first label file's dimensions
    class_counts: Optional[numpy.ndarray] = None

    # Count class occurrences across all label arrays
    for labels_arr in tqdm(labels_iterable):
        # Initialize class counts array on the first iteration
        if class_counts is None:
            labels_height, labels_width = labels_arr.shape
            class_counts = numpy.zeros(shape=(labels_height, labels_width, n_classes), dtype=numpy.uint64)

        # Accumulate class counts for each pixel
        for class_idx in range(n_classes):
            class_counts[:, :, class_idx] += (labels_arr == class_idx).astype(numpy.uint64)

    # Total valid observations per pixel avoiding ZeroDivisionError
    # Shape (height, width, 1)
    total_counts = class_counts.sum(axis=2, keepdims=True)
    zero_mask = (total_counts == 0).squeeze(axis=2)  # Shape: (height, width)
    total_counts[zero_mask] = 1  # Temporarily set to 1 to avoid division by zero

    # Compute class priors as normalized counts.
    # Shape: (height, width, n_classes)
    class_priors = class_counts / total_counts

    # Fill pixels with zero valid observations
    if fill_value is not None:
        # Specified fill value
        class_priors[zero_mask] = fill_value
    else:
        # Uniform distribution
        class_priors[zero_mask] = 1.0 / n_classes

    return class_priors


# DATASET HANDLER --------------------------------------------------
def FRF_Duck_handler():
    """Handler for computing class priors for the FRF Duck dataset."""
    # FRF Duck data
    output_file = Path("assets/annotations/FRF_Duck/class_priors.npy")
    mask_loader = DuckMaskDataset(
        filepaths=list(Path("assets/annotations/FRF_Duck/train/labels").glob("*.png")),
        use_train_classes=False
    )

    # Compute class priors
    class_priors = compute_class_priors(
        labels_iterable=mask_loader,
        n_classes=len(mask_loader.get_classes())
    )

    # Save class priors array
    output_file.parent.mkdir(parents=True, exist_ok=True)
    numpy.save(file=output_file, arr=class_priors)
    print(f"Class prior probabilities saved to: '{output_file}'")

    # Save figure
    figure_file = output_file.with_suffix(".png")
    save_priors_figure(
        class_priors=class_priors,
        class_names=mask_loader.get_classes(),
        output_file=figure_file
    )

# python -m src.handlers.create_prior_probabilities
if __name__ == "__main__":
    FRF_Duck_handler()