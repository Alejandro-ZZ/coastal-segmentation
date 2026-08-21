# Coastal Segmentation

This repository contains the code, trained artifacts, and experiment assets for pixel-wise semantic segmentation of coastal imagery. The primary workflow trains a **Random Forest** on local color neighborhoods and optionally refines class probabilities with spatial **Bayesian priors** or a dense **Conditional Random Field** (CRF). Reference notebooks provide the Segment Anything Model (**SAM**), **U-Net**, and **ResUNet** baselines used in the study.

The command-line interface in `src/handlers/` supports reproducible feature extraction, model fitting, spatial-prior generation, and batch inference. Run every command from the repository root.

## 📋 Table of Contents

* [📂 Repository Layout](#-repository-layout)
* [⬇️ Installation](#️-installation)
* [📥 Duck Data Retrieval](#-duck-data-retrieval)
* [📊 Prior Probabilities](#-prior-probabilities)
* [⚠️ Expected data](#️-expected-data)
* [🚀 Usage](#-usage)
    * [⚙️ Train processor](#️-train-processor)
    * [📷 Segment images](#-segment-images)
* [🧠 Deep Learning Baselines](#-deep-learning-baselines)
* [🐍 Python Users](#-python-users)


## 📂 Repository Layout

```text
coastal-segmentation/
├── .gitignore                              # Excluded files and directories
├── README.md                               # Installation, data setup, CLI, and Python API reference
├── requirements.txt                        # Python dependencies
│
├── assets/                                 # Annotation metadata, masks, and model settings
│   ├── optimized_RF_params.json            # All tuned Random Forest hyperparameters
│   ├── RF_params_0NN.json                  # Optimal Random Forest params for single pixel features
│   ├── RF_params_8NN.json                  # Optimal Random Forest params for 8-neighbor features
│   ├── RF_params_24NN.json                 # Optimal Random Forest params for 24-neighbor features
│   └── annotations/
│       ├── FRF_Duck/                       # Duck dataset data
│       │   ├── class_priors.png            # Visualization of the Duck spatial class priors
│       │   ├── classes.txt                 # Original Duck class names
│       │   ├── small_classes.txt           # Four-class remapping names
│       │   ├── small_classes_config.json   # Ordered classes and palette for the remapped labels
│       │   ├── train_config_resunet.json   # Published ResUNet training configuration
│       │   ├── train_image_files.txt       # Selected image filenames for RF-based training
│       │   ├── train_metadata.csv          # Image-level metadata for the Duck training set
│       │   └── train_metadata_code.py      # Metadata-generation and file selection code
│       └── PHCO/                           # Pehuen Co rectified-image annotations
│           ├── classes_config.json         # Ordered PHCO classes and visualization palette
│           ├── points_rectify_split.csv    # Point labels, rectified coordinates, and data splits
│           └── roi_rectify_mask.png        # Valid-pixel ROI for rectified PHCO imagery
│
├── results/
│   ├── datasets/                           # Generated point samples dataset
│   │   └── ...                       
│   └── models/                             # Trained model artifacts
│       └── ...                       
│
└── src/                                    # All source code, CLI commands, and notebooks
  ├── processors.py                         # Random-Forest-based pipeline 
  ├── utils.py                              # General function helpers
  ├── handlers/                             # Module-based CLI commands
  │   ├── create_priors.py                  
  │   ├── prepare_duck_assets.py            
  │   ├── segment_images.py               
  │   └── train_processor.py              
  └── notebooks/                            # Colab-oriented deep-learning baseline workflows
    ├── ResUnet_workflow.ipynb          
    ├── SAM_workflow.ipynb              
    └── Unet_workflow.ipynb             
```

Downloaded Duck images, extracted labels, trained models, predictions, and local experiment files are intentionally not versioned. Use the [Duck setup command](#-duck-data-retrieval) to populate the required image assets.


## ⬇️ Installation

### Computational requirements

* **Minimum (Random Forest pipelines):** Executable on standard consumer laptops (Dual-core CPU, 8 GB RAM). Training is highly parallelized.

* **Recommended (Deep learning):** NVIDIA GPU with $\ge$ 12 GB VRAM (for SAM ViT-H inference and dense UNet/ResUNet training). Experiments were conducted on a free Google Colab T4 GPU (16 GB VRAM).

> **Note:** Random Forest feature extraction and inference operate on every image pixel at prediction time. Memory and runtime therefore increase with both image resolution and neighborhood size: a $D24$ feature vector contains 75 channel values per pixel before classifier computation. The CPU pipeline uses all available cores when the serialized Random Forest was trained with `n_jobs=-1`.

### Dependencies

* We recommend using a virtual environment to isolate the project dependencies. The project requires Python 3.11 or higher.

* When cloning the repository, use the `--depth 1` option to create a shallow clone with only the latest commit, which is sufficient for running the code.

```bash
# Clone the repository
git clone --depth 1 https://github.com/Alejandro-ZZ/coastal-segmentation.git

# Create and activate a virtual environment 
# (conda is optional but recommended)
conda create -n coast-segment python=3.11
conda activate coast-segment

# Install the required dependencies
pip install --upgrade pip
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

The Bayesian refinement method requires same-size, aligned integer label masks. For FRF Duck, `--duck-labels` applies the project’s label resizing and six-to-four class remapping before computing a $(H, W, C)$ prior tensor.

> **NOTE:** The below multi-line command is for Linux/macOS. For Windows users, break the command into multiple lines with `^` instead of `\` at the end of each line.

```bash
python -m src.handlers.create_priors \
  --labels assets/annotations/FRF_Duck/train/labels \
  --pattern "*.png" \
  --duck-labels \
  --n-classes 4 \
  --output assets/annotations/FRF_Duck/class_priors.npy
```

Custom mask generator can be created by adding a subclass of `BaseMaskGenerator` to adapt mask processing for other datasets. See code details in `src/handlers/create_priors.py` and the `ImageMaskGenerator` and `DuckMaskGenerator` classes.

## ⚠️ Expected data

### Class configuration

The `train_processor` recieves a `--classes-config` argument. This must be a JSON object with ordered `classes` and optional RGB `palette` arrays. Array position defines the integer label used by the model, so both order and length are part of the model contract. Example:

```json
{
  "classes": [
        "background", 
        "sand", 
        "vegetation", 
        "lag"
    ],
  
  "palette": [
        [1, 1, 255], 
        [255, 255, 1], 
        [1, 255, 1], 
        [255, 1, 1]
    ]
}
```

### CSV annotations

The expected training and feature-extraction annotation files use one row per annotated pixel:

| Column | Type | Meaning |
| --- | --- | --- |
| `ImageFile` | string | Image filename relative to the handler input `--images` diretory |
| `Cx` | integer | Zero-based horizontal pixel coordinate |
| `Cy` | integer | Zero-based vertical pixel coordinate |
| `Class` | string | Class name; required for training and must be present in `classes_config.json` |
| `Split` | string | Optional partition label used by `train_processor --split` |

Coordinates must reference valid pixels in their corresponding images. `train_processor` requires integer coordinates; use `--coerce-coordinates` only for known rectified coordinates that require conversion.


### Spatial mask and priors

For standard prior construction, labels must be two-dimensional integer images with values from `0` through `n_classes - 1`, all with identical dimensions. The saved `class_priors.npy` has shape `(height, width, n_classes)`, values in `[0, 1]`, and class probabilities summing to one for every pixel.

An optional ROI mask passed to `segment_images --roi-mask` is a binary image. Only nonzero mask pixels are classified. Output labels outside the ROI use the processor’s ignore index and probabilities are uniform.

## 🚀 Usage

> **NOTE:** The below multi-line commands are for Linux/macOS. For Windows users, break the command into multiple lines with `^` instead of `\` at the end of each line.

### ⚙️ Train processor

* The `train_processor` handler fits the Random-Forest-based pipeline directly from CSV point annotations. 

* Model serialization uses `joblib`. Load processors with compatible Python and scikit-learn versions.

#### Inputs

| Parameter | Description |
|---|---|
| --images | Directory containing training images. |
| --annotations | CSV file with `ImageFile`, `Cx`, `Cy`, and `Class` columns. |
| --classes-config | JSON containing classes and/or palette arrays. |
| --neighbors | Neighborhood size. Possible values: 0, 8, 24. |
| --colorspace | Preprocessing color space (default: RGB). For possible values see [skimage.color.convert_colorspace](https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.convert_colorspace)  |
| --output | Output artifacts path: PKL and JSON files. |
| --split | Optional `Split`-column value used to select training rows. See below PHCO example. |
| --coerce-coordinates | Convert `Cx` and `Cy` values to integers before training. |
| --classifier-params | JSON object of scikit-learn RandomForestClassifier parameters. |


* See detailed command-line help with: 

```bash
python -m src.handlers.train_processor --help
```

* The parameters in `assets/RF_params_24NN.json` file are the found optimal values of Ranfom Forest for 24 nearest neighbors. The fixed seed and parallel configuration in the legacy paper handler are `random_state=23` and `n_jobs=-1`; include these values in a copied parameter JSON when a fully identical retraining configuration is required.

#### Outputs

* Serialized PKL file with the `SegmentationProcessor` instance. 

* Structured JSON file with the complete processor metadata.


#### FRF Duck - 24NN

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

#### Pehuen Co (PHCO) - 24NN

* `--split train`: selects only the training samples (i.e., the rows with `Split` column value `train`) from the CSV point annotations. 

* `--coerce-coordinates`: converts the rectified floating-point coordinates (columns `Cx` and `Cy`) before fitting.

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


### 📷 Segment images

#### Inputs

| Parameter | Description |
|---|---|
| --images | Directory containing input images. |
| --pattern | Glob pattern used to select input images (default: *.jpg). |
| --model | Trained `SegmentationProcessor` PKL file. |
| --output | Directory for compressed predictions and color-label PNG files. |
| --refine | Optional spatial refinement. Possible values: `none` (default), `bayes`, `crf` |
| --roi-mask | Optional binary image file mask. Only nonzero pixels are segmented. |
| --class-priors | Spatial class-prior NPY file. Required for `--refine bayes`. |

#### Outputs

For every source image, the segmentation command stores:

* A compressed NumPy array (NPZ file) saved as `{image_stem}.npz` under the given `--output` directory. The compressed NPZ contains:

    - `classes`: ordered class names.
    - `image`: source RGB image.
    - `class_proba`: $(H, W, C)$ posterior probabilities.
    - `labels`: $(H, W)$ integer labels.
    - `color_labels`: RGB palette visualization.
    - `overlay`: source image blended with the palette visualization.

* A colorized PNG segmented image, saved as `{image_stem}.png` under the `--output/color_labels` directory.

#### Examples

1. Raw inference (no refinement)

```bash
python -m src.handlers.segment_images \
  --images assets/annotations/FRF_Duck/test_c1/images \
  --pattern "*.jpg" \
  --model results/models/FRF_Duck_24NN_SegmentationProcessor.pkl \
  --output results/predictions/FRF_Duck/test_c1/24NN/RF
```

2. Inference with spatial Bayesian prior

```bash
python -m src.handlers.segment_images \
  --images assets/annotations/FRF_Duck/test_c1/images \
  --pattern "*.jpg" \
  --model results/models/FRF_Duck_24NN_SegmentationProcessor.pkl \
  --output results/predictions/FRF_Duck/test_c1/24NN/RF-Bayes \
  --refine bayes \
  --class-priors assets/annotations/FRF_Duck/class_priors.npy \
```

3. Inference with dense CRF

```bash
python -m src.handlers.segment_images \
  --images assets/annotations/FRF_Duck/test_c1/images \
  --pattern "*.jpg" \
  --model results/models/FRF_Duck_24NN_SegmentationProcessor.pkl \
  --output results/predictions/FRF_Duck/test_c1/24NN/RF-CRF \
  --refine crf
```

## 🧠 Deep Learning Baselines

The reference workflows for SAM and dense supervised Unet models are notebooks developed for a [Google Colab](https://colab.research.google.com/) T4 GPU and may also run locally with a compatible CUDA/TensorFlow or PyTorch installation. Use the downloaded ResUNet directory described in [Duck Data Retrival](#-duck-data-retrieval) when loading the published TensorFlow model.

### Notebooks

#### [U-Net](src/notebooks/Unet_workflow.ipynb)

This is the fully supervised Duck baseline. It pairs JPEG images with matching `*_label.png` masks, remaps the original six labels to four classes (`background`, `sand`, `vegetation`, `lag`), and trains a 512 x 512 U-Net with sparse categorical cross-entropy. The notebook uses a training-validation split, image resizing, GPU mixed precision, checkpointing, early stopping, and learning-rate reduction.

The core construction and single-image inference flow are:

```python
import tensorflow as tf
from pathlib import Path

# TODO: Include the CustomMacroF1 class definition (implemented in the notebook)

# Model checkpoint path
MODEL_CHECKPOINT_FILE = "results/models/FRF_Duck_Unet.keras" 

# Image file to segment
image_file = Path("assets/annotations/FRF_Duck/test_c1/images/FRF_c1_snap_20161107160000_EBG.jpg")

# Load the model
model = tf.keras.models.load_model(
    filepath=MODEL_CHECKPOINT_FILE,
    custom_objects={"CustomMacroF1": CustomMacroF1}
)

# Make inference on a single image
results = process_and_segment(image_file, model)

# 2-D integer array of shape (height, width)
pred_mask = results["labels"]
```

* Check the ``CustomMacroF1`` class in the ``✨ Definitions`` section of the notebook for the custom metric used during training.

* The `process_and_segment` function restores predictions to the source resolution and produces labels and palette visualizations. 

* The `results` dictionary contains the original image (`image`), predicted mask (`labels`), colorized labels (`color_labels`), low-resolution image (`low_image`) and low-resolution mask (`low_labels`).

* Training section saves a Keras model, history, TensorBoard logs, and NPZ/PNG prediction artifacts. The notebook is arranged for Google Colab and benefits substantially from a GPU.


#### [ResUNet](src/notebooks/ResUnet_workflow.ipynb)

This notebook evaluates the published pretrained Duck ResUNet rather than training an architecture from scratch. It loads the extracted model with a custom `ResnetCustomMacroF1` metric, standardizes each image, uses four-class one-hot masks, and evaluates 512 x 512 batches.

```python
import tf_keras
from pathlib import Path

# TODO: Include the ResnetCustomMacroF1 class definition (implemented in the notebook)

# Model checkpoint path
model_path = "results/models/FRF_mar15_remap_fullmodel_model"

# Test image and mask files
data_path = Path("assets/annotations/FRF_Duck/test_c1")
image_file = data_path / "images/FRF_c1_snap_20161107160000_EBG.jpg"
mask_file = image_file.with_name(image_file.stem + "_label.png")


# Preprocess input image
image, one_hot_mask = resnet_process_file(image_file, mask_file)

# Load the pretrained model
model = tf_keras.models.load_model(
  model_path,
  custom_objects={"ResnetCustomMacroF1": ResnetCustomMacroF1},
)

# Predict class probabilities
# Float array of shape (1, height, width, n_classes)
pred_proba = model.predict(numpy.expand_dims(image, axis=0))

# Predicted class labels
# 2-D integer array of shape (height, width)
pred_mask = numpy.argmax(pred_proba.squeeze(), axis=-1)
```

* See the ``ResnetCustomMacroF1`` class in the ``✨ Definitions`` section of the notebook for the custom metric used.

* It reports confusion matrices plus per-class precision, recall, F1, and Jaccard/IoU metrics, exporting split-level JSON reports. Download and extract the published ResUNet weights with the [Duck setup command](#-duck-data-retrieval) before running it.


#### [SAM](src/notebooks/SAM_workflow.ipynb)

This is a point-prompted, multi-class SAM workflow for rectified PHCO images. It installs Meta's Segment Anything package, downloads the ViT-H checkpoint, caches each image embedding, subsamples labeled points by class, and can include background points as negative prompts. Annotation tables must provide image name, rectified x/y coordinates, class name, and optionally a split. The ROI mask removes predictions outside valid rectified pixels.

```python
# Initialize the SAM predictor
sam_predictor = initialize_sam(
  model_type="vit_h", 
  checkpoint_path=".../sam_vit_h_4b8939.pth", 
  device="cuda"
)

# Create the SAM pipeline wrapper
model = SparsePointsSamPredictor(
  predictor=sam_predictor, 
  device="cuda"
)

# Set the input image and prompt points
model.set_input(
  image,          # Image array as (height, width, 3)
  input_points,   # Integer array of (x, y) points as (n_points, 2)
  target_labels,  # Integer array of class labels as (n_points, )
  sorted_classes  # Index-sorted class names, matching the target labels values
)

# Run inference with optional background prompts
results = model.make_inference(use_background_prompts=True)

# 2-D integer array of shape (height, width)
pred_mask = results["labels"]
```

* The ``results`` contain a per-class logit map (`logits`) and a 2-D label image (`labels`). 

* The notebook also writes palette PNGs and prompt-selection diagnostics. ViT-H is GPU-oriented, and limiting prompts to roughly 11,000 per image avoids excessive VRAM use.


## 🐍 Python Users

* The core `SegmentationProcessor` class encapsulates the complete classifier-based and feature extraction process. See the complete documentation in `src/processors.py` for details on the class and its methods. 

### Create instace

* You can use any scikit-learn-like classifier, supporting the `fit`, `predict`, and `predict_proba` methods. 

* `colors` parameter is optional. If not provided, the processor will use a default color palette for visualization.

```python
from src.processors import SegmentationProcessor

processor = SegmentationProcessor(
    n_neighbors=24,
    classifier=RandomForestClassifier(),
    classes=["class1", "class2", "class3"],
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  
)
```

### Preprocessing

* The processor have a `preprocessing` attribute of type `PreprocessingBlock` that can be used to apply color space conversion and other preprocessing steps to images before feature extraction. 

* **Colorspace:** set the color space from which the features are extracted. The default is RGB, but you can change it to other color spaces supported by [skimage.color.convert_colorspace](https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.convert_colorspace).

```python
processor.preprocessing.set_colorspace("YIQ")
```

* **Gaussian smoothing:** apply Gaussian filter to the images before feature extraction. A value of 0 (default) means no smoothing.

```python
processor.preprocessing.set_gaussian_sigma(1.0)
```

### Postprocessing

* The processor have a `postprocessing` attribute of type `PostprocessingBlock` that can be used to apply custom postprocessing steps to the predictions.

* **Background color:** set the background color for the output color label images. The default is black. This relevant when segmenting images using a ROI mask.

```python
processor.postprocessing.set_background_color([0, 0, 0])
```

* **Majority filter:** footprint to apply a majority filter to the predicted labels to remove small isolated regions. The neighborhood footprint must be expressed as a 2-D array of 1’s and 0’s. A value of `None` (default) means no filtering. See more details in [skimage.filters.rank.majority](https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.majority)

```python
from skimage.morphology import disk

processor.postprocessing.set_majority_filter_size(
    footprint=disk(3)
)
```

### Train

* When training the processor, the results are also populated in the `training_metadata` class attribute. See below evaluation section for details on the structure of the results.

```python
import pandas
from pathlib import Path

# Images path
img_path = Path("assets/annotations/FRF_Duck/train/images")

# Dataset
dataset_file = "results/datasets/FRF_Duck/dataset_points.csv"
dataset = pandas.read_csv(dataset_file)

# Fit and evaluate the processor
train_results = processor.evaluate_classifier(
    images_path=img_path,
    annotations_data=dataset,
    do_train=True
)
pprint(train_results)
print("\n")

# Save the trained processor to a file
processor.save("path/to/save/processor.pkl")

# Get the complete metadata of the trained processor
metadata = processor.get_metadata()
pprint(metadata)
```

### Evaluate

* If you have a validation/test subset, you can evaluate the processor after training using `do_train=False` and passing the validation/test dataset to `evaluate_classifier`.

```python
eval_results = processor.evaluate_classifier(
    images_path=img_path,
    annotations_data=dataset,
    do_train=False
)
pprint(eval_results)
```

Evaluation results is a dictionary containing the following keys:  

* ``date_UTC``: evaluation timestamp in UTC (ISO format).

* ``data``: summary of the input data with keys:

    * ``n_samples``: number of annotated samples (pixels)
    * ``n_features``: number of features per sample (depends on the neighborhood size)
    * ``label_counts``: dictionary with the count of samples per class label

*   ``timings``: timing information for each step (feature extraction, training, prediction, total)

*   ``metrics``: classification report as returned by `sklearn.metrics.classification_report()`.

### Segment image

* You can segment a single image using the trained processor.

```python
import skimage

# Input image
img_file = "assets/annotations/FRF_Duck/test_c1/images/FRF_c1_snap_20161107160000_EBG.jpg"
img = skimage.io.imread(img_file)

# Segment the image with optional refinement
results = processor.segment_image(
    rgb_image=img, 
    refine="crf"
)
```

Predicted results data is a dictionary containing the following keys:
            
*   ``"classes"``: List of class names corresponding to the predicted labels.

*   ``"image"``: The original input RGB image.

*   ``"class_proba"``: predicted class probabilities. 3D array as `(height, width, n_classes)`

*   ``"labels"``: predicted class labels. 2D integer array as `(height, width)`.

*   ``"color_labels"``: color-coded image labels. 3D uint8 RGB array as `(height, width, 3)`.

*   ``"overlay"``: overlay visualization. 3D uint8 RGB array of shape `(height, width, 3)`.