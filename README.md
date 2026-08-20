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
* [Python Users](#-python-users)



## ⬇️ Installation

1. Clone the repository.

```bash
git clone --depth 1 https://github.com/Alejandro-ZZ/coastal-segmentation.git
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

> **NOTE:** The below multi-line commands are for Linux/macOS. For Windows users, break the command into multiple lines with `^` instead of `\` at the end of each line.

### ⚙️ Train processor

* The `train_processor` handler fits the Random-Forest-based pipeline directly from CSV point annotations. 

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

* A compressed NumPy array (`.npz`) of predicted class labels for each pixel, saved as `{image_stem}.npz` in the given `{--output}/` directory. 

* A colorized PNG of the predictions, saved as `{image_stem}.png` in the `{--output}/color_labels` directory.

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

...

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