# Coastal Segmentation

This repository contains the code, trained artifacts, and experiment assets for pixel-wise semantic segmentation of coastal imagery. The primary workflow trains a Random Forest on local color neighborhoods and optionally refines class probabilities with spatial Bayesian priors or a dense Conditional Random Field (CRF). Reference notebooks provide the Segment Anything Model (SAM), U-Net, and ResUNet baselines used in the study.

## Table of Contents

* [Installation](#️-installation)
* [Duck Data Retrieval](#-duck-data-retrieval)
* [Prior Probabilities](#-prior-probabilities)
* [Usage](#-usage)
    * [Train processor](#️-train-processor)
    * [Segment images](#-segment-images)
* [Deep Learning Baselines](#-deep-learning-baselines)



## ⬇️ Installation

1. Clone the repository.

TODO: Update repo URL
```bash
git clone --depth 1 https://github.com/yourusername/coastal-segmentation.git
```

The `--depth 1` option creates a shallow clone with only the latest commit, which is sufficient for running the code.

2. Create and activate a virtual environment (optional but recommended using conda):

```bash
conda create -n coast-segment python=3.11
conda activate coast-segment
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```



## 📥 Duck Data Retrieval

The Duck dataset was presetned on an [Springer Conference paper](https://link.springer.com/chapter/10.1007/978-3-032-15477-4_55) and is a collection of coastal images used for validating the segmentation pipelines. The complete assets are available in the [Zenodo records](https://zenodo.org/records/7075342):

| Resource | Zenodo record | Required extracted directory |
| --- | --- | --- |
| Training data | [training_data.zip](https://zenodo.org/records/7075342/files/training_data.zip?download=1) | `assets/annotations/FRF_Duck/train/` |
| Test data, camera C1 | [test_data_c1.zip](https://zenodo.org/records/7075342/files/test_data_c1.zip?download=1) | `assets/annotations/FRF_Duck/test_c1/` |
| Test data, camera C6 | [test_data_c6.zip](https://zenodo.org/records/7075342/files/test_data_c6.zip?download=1) | `assets/annotations/FRF_Duck/test_c6/` |
| ResUNet weights | [model.zip](https://zenodo.org/records/7075342/files/model.zip?download=1) | `model/FRF_mar15_remap_fullmodel_model/` |

### Automated Setup

Use the platform-independent Python command below from the repository root. It downloads the Zenodo archives, extracts them, installs each directory in the expected project location, and downloads the ResUNet model:

```bash
python -m src.handlers.prepare_duck_assets
```

The command also prepares the segmentation assets before any model command is run:

- Cleans every JPEG label mask and saves a corresponding PNG mask.
- Resizes only the training images and their matching masks to $(2048, 2448)$. RGB images use anti-aliased resampling; masks use nearest-neighbor resampling.
- Test C1 and C6 images are not resized.

By default, the command refuses to replace existing asset directories. On a fresh checkout no extra option is needed. To deliberately replace existing downloaded assets, use:

```bash
python -m src.handlers.prepare_duck_assets --overwrite
```

Inspect the planned actions without changing files with:

```bash
python -m src.handlers.prepare_duck_assets --dry-run
```

Use `--assets` to prepare a subset, such as `--assets train test_c1 test_c6`; `--download-dir downloads` retains the ZIP files for reuse; `--skip-download` processes ZIP files already in that directory. For extra help, run:

```bash
python -m src.handlers.prepare_duck_assets --help
```

TensorFlow should load the extracted model from `results/models/FRF_mar15_remap_fullmodel_model`. Do not move the Random Forest model files away from `results/models/`; the documented experiments expect that location.



## 📊 Prior Probabilities

...

## 🚀 Usage

### ⚙️ Train processor

The below handler fits the Random-Forest-based pipeline directly from CSV point annotations, saving a serialized `.pkl` processor and matching `.json` metadata. The parameters in `assets/RF_params_24NN.json` are selected by neighborhood size. The fixed seed and parallel configuration in the legacy paper handler are `random_state=23` and `n_jobs=-1`; include these values in a copied parameter JSON when a fully identical retraining configuration is required.

```bash
python -m src.handlers.train_processor \
  --images assets/annotations/FRF_Duck/train/images \
  --annotations results/datasets/FRF_Duck/dataset_points.csv \
  --classes-config assets/annotations/FRF_Duck/small_classes_config.json \
  --neighbors 24 \
  --colorspace YIQ \
  --classifier-params assets/RF_params_24NN.json \
  --output results/models/FRF_Duck_24NN
```

For Pehuen Co (PHCO), select the training partition and convert rectified floating-point coordinates before fitting:

```bash
python -m src.handlers.train_processor \
  --images assets/annotations/PHCO/train_points \
  --annotations assets/annotations/PHCO/points_rectify_split.csv \
  --split train \
  --coerce-coordinates \
  --classes-config assets/annotations/PHCO/classes_config.json \
  --neighbors 24 \
  --colorspace YIQ \
  --classifier-params assets/RF_params_24NN.json \
  --output results/models/PHCO_24NN
```

> **NOTE:** for Windows OS break the command into multiple lines with `^` instead of `\` at the end of each line, or run it in a single line without line breaks.

The core `SegmentationProcessor` class encapsulates the complete Random Forest and neighboring process. See the complete documentation in `src/processors.py` for details on the class and its methods. An example usage is provided below:

```python
import pandas
from pprint import pprint
from src.processors import SegmentationProcessor

# Dummy dataset
dataset = pandas.DataFrame({
    "ImageFile": ["image1.jpg", "image2.jpg"],
    "Cx": [100, 150],
    "Cy": [200, 250],
    "Class": ["class1", "class2"]
})

# Create a SegmentationProcessor instance with the desired configuration
processor = SegmentationProcessor(
    n_neighbors=24,
    colorspace="YIQ",
    classes=["class1", "class2", "class3"],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # RGB values for each class
    classifier_params={"n_estimators": 100, "max_depth": None},
)

# Train the processor using the dataset and images path
processor.train(images_path="path/to/images", annotations_data=dataset)

# Save the trained processor to a file
processor.save("path/to/save/processor.pkl")

# Get the complete metadata of the trained processor
metadata = processor.get_metadata()
pprint(metadata)
```

### 📷 Segment images

1. Raw inference (no refinement):

```bash
python -m src.cli.segment_images \
  --images assets/annotations/FRF_Duck/test_c1/images \
  --pattern "*.jpg" \
  --model results/models/FRF_Duck_24NN_SegmentationProcessor.pkl \
  --output results/predictions/FRF_Duck/test_c1/24NN/RF
```

2. Inference with spatial Bayesian prior:

```bash
python -m src.cli.segment_images \
  --images assets/annotations/FRF_Duck/test_c1/images \
  --pattern "*.jpg" \
  --model results/models/FRF_Duck_24NN_SegmentationProcessor.pkl \
  --output results/predictions/FRF_Duck/test_c1/24NN/RF-Bayes \
  --refine bayes \
  --class-priors assets/annotations/FRF_Duck/class_priors.npy \
```

3. Inference with dense CRF:

```bash
python -m src.cli.segment_images \
  --images assets/annotations/FRF_Duck/test_c1/images \
  --pattern "*.jpg" \
  --model results/models/FRF_Duck_24NN_SegmentationProcessor.pkl \
  --output results/predictions/FRF_Duck/test_c1/24NN/RF-CRF \
  --refine crf
```

The handlers creates two outputs:

* A compressed NumPy array (`.npz`) of predicted class labels for each pixel, saved as `{image_stem}.npz` in the `{--output}/` directory.

* A colorized PNG of the predicted labels, saved as `{image_stem}.png` in the `{--output}/color_labels` directory.

## 🧠 Deep Learning Baselines

...