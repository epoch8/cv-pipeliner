# CV-pipeliner

`cv_pipeliner` is a small Python library for building computer vision workflows around common data objects. It helps you describe images and annotations, convert datasets between formats, run detection/classification/keypoint/embedder models, evaluate predictions, and visualize results.

The package is centered around two data classes:

- `ImageData`: one image, optional image-level label/keypoints/mask, and a list of bounding boxes.
- `BboxData`: one object inside an image, with coordinates, label, scores, keypoints, masks, nested boxes, and crop helpers.

For a longer runnable walkthrough, see [`docs/getting_started.ipynb`](docs/getting_started.ipynb).

## Installation

From this repository:

```bash
cd cv-pipeliner
poetry install
```

Optional model backends are exposed as Poetry extras:

```bash
poetry install --extras tensorflow
poetry install --extras torch
```

If you install the package with `pip` from a local checkout:

```bash
pip install .
pip install ".[tensorflow]"
pip install ".[torch]"
```

Python `>=3.9,<3.14` is supported. Some optional ML backends currently support narrower Python ranges; check [`pyproject.toml`](pyproject.toml) before choosing an environment.

## What Is Included

`cv_pipeliner` exports the most commonly used APIs from the top-level package:

- Data objects: `ImageData`, `BboxData`.
- Batch generators: `BatchGeneratorImageData`, `BatchGeneratorBboxData`.
- Annotation converters: `JSONDataConverter`, `COCODataConverter`, `YOLODataConverter`, `YOLOMasksDataConverter`, `SuperviselyDataConverter`.
- Model specs and inferencers: `YOLOv8_ModelSpec`, `YOLOv5_ModelSpec`, TensorFlow/PyTorch model specs, `DetectionInferencer`, `ClassificationInferencer`, `KeypointsRegressorInferencer`, `PipelineInferencer`, `PipelineModelSpec`.
- Metrics: `get_df_detection_metrics`, `get_df_classification_metrics`, `get_df_pipeline_metrics`.
- Visualization and image utilities: `visualize_image_data`, `visualize_image_data_matching_side_by_side`, resize/crop/rotate helpers, non-max suppression, and image concatenation helpers.

## Core Data Model

Create image annotations with `ImageData` and `BboxData`:

```python
from cv_pipeliner import BboxData, ImageData

image_data = ImageData(
    image_path="images/example.jpg",
    label="scene-label",
    bboxes_data=[
        BboxData(
            xmin=25,
            ymin=40,
            xmax=180,
            ymax=210,
            label="object",
            detection_score=0.97,
            keypoints=[(60, 75), (120, 150)],
            mask=[[(25, 40), (180, 40), (180, 210), (25, 210)]],
        )
    ],
)
```

`ImageData` can be built from an image path, bytes, a PIL image, or a NumPy array. When an in-memory image is provided, `meta_width` and `meta_height` are inferred automatically. When only a path is provided, the size is read lazily when needed.

```python
image = image_data.open_image(inplace=True)
width, height = image_data.get_image_size()

bbox = image_data.bboxes_data[0]
crop = bbox.open_cropped_image()
crop_data = bbox.open_cropped_image(return_as_image_data=True)
```

When `ImageData` owns source fields such as `image_path`, `image`, `meta_width`, and `meta_height`, those fields are propagated to nested `BboxData` objects. This keeps crops, coordinate normalization, and nested annotations consistent.

## ImageData Transformations

`cv_pipeliner.utils.images_datas` contains helpers that transform the whole `ImageData` object, not only the raw image. When you resize, crop, rotate, or apply a perspective transform, the related annotation fields are updated together with the image: bounding boxes, keypoints, masks, nested `additional_bboxes_data`, metadata, and cached crops where applicable.

The most commonly used helpers are exported from the top-level package:

```python
from cv_pipeliner import (
    apply_perspective_transform_to_image_data,
    crop_image_data,
    flatten_additional_bboxes_data_in_image_data,
    non_max_suppression_image_data,
    resize_image_data,
    rotate_image_data,
    thumbnail_image_data,
)

resized = resize_image_data(image_data, size=(640, 480))
rotated = rotate_image_data(image_data, angle=15)
thumbnail = thumbnail_image_data(image_data, size=320)

cropped = crop_image_data(
    image_data,
    xmin=100,
    ymin=50,
    xmax=500,
    ymax=400,
    allow_negative_and_large_coords=False,
    remove_bad_coords=True,
)

filtered = non_max_suppression_image_data(
    image_data,
    iou=0.5,
    score_threshold=0.25,
)
flat = flatten_additional_bboxes_data_in_image_data(image_data)
```

These helpers return transformed copies of `ImageData`, so the original annotation object can be reused for other experiments. This is useful for preparing training data, building crop-based pipelines, visualizing augmented samples, or postprocessing model predictions without manually recalculating every coordinate field.

## Annotation Conversion

Converters translate between external annotation formats and `ImageData`.

### YOLO

```python
from cv_pipeliner import BboxData, ImageData, YOLODataConverter

converter = YOLODataConverter(class_names=["cat", "dog"])

image_data = ImageData(
    image_path="images/cat.jpg",
    bboxes_data=[BboxData(xmin=10, ymin=20, xmax=100, ymax=160, label="cat")],
)

yolo_lines = converter.get_annot_from_image_data(image_data)
restored = converter.get_image_data_from_annot(
    image_path="images/cat.jpg",
    annot=yolo_lines,
)
```

Use `YOLOMasksDataConverter` for polygon mask annotations in YOLO segmentation format.

### COCO

```python
from cv_pipeliner import COCODataConverter

converter = COCODataConverter()
image_data = converter.get_image_data_from_annot(
    image_path="images/000000000009.jpg",
    annot="annotations/instances_train2017.json",
)
```

`COCODataConverter` reads COCO boxes and segmentation polygons into `BboxData` objects.

## Labeling Tool Integrations

`ImageData` can also be converted to and from common annotation and dataset inspection tools.

### FiftyOne

`FifyOneSession` converts `ImageData`, `BboxData`, and `ImageDataMatching` objects into FiftyOne samples, detections, keypoints, and error-analysis views. It also converts FiftyOne samples back into `ImageData`.

```python
from cv_pipeliner import FifyOneSession

fo_session = FifyOneSession(database_dir=".fiftyone")

sample = fo_session.convert_image_data_to_fo_sample(
    image_data,
    fo_detections_label="ground_truth",
    fo_classification_label="image_label",
    fo_keypoints_label="keypoints",
    include_additional_bboxes_data=True,
)

restored = fo_session.convert_sample_to_image_data(
    sample,
    fo_detections_label="ground_truth",
    fo_classification_label="image_label",
    fo_keypoints_label="keypoints",
)
```

The integration can also represent matching results as FiftyOne detections, which is useful for browsing TP/FP/FN cases after detection or pipeline evaluation.

### Label Studio

`cv_pipeliner.utils.label_studio` converts `ImageData` to Label Studio annotation dictionaries and parses Label Studio results back into `ImageData`. It supports image-level choices, rectangle labels, polygon masks, keypoints, and relations between boxes and keypoints.

```python
from cv_pipeliner.utils.label_studio import (
    convert_annotation_to_image_data,
    convert_image_data_to_annotation,
)

annotation = convert_image_data_to_annotation(
    image_data,
    to_name="image",
    bboxes_from_name="bbox",
    label_from_name="label",
    keypoints_from_name="keypoint",
    keypoints_labels=["left", "right"],
    mask_from_name="mask",
)

restored = convert_annotation_to_image_data(
    annotation,
    bboxes_from_name="bbox",
    label_from_name="label",
    keypoints_from_name="keypoint",
    keypoints_labels=["left", "right"],
    mask_from_name="mask",
    image_path="images/example.jpg",
)
```

## Batch Generators

Batch generators open image/crop data on demand and clear it after each batch to reduce memory pressure.

```python
from cv_pipeliner import BatchGeneratorBboxData, BatchGeneratorImageData

images_data_gen = BatchGeneratorImageData(data=[image_data], batch_size=8)
bboxes_data_gen = BatchGeneratorBboxData(
    data=[image_data.bboxes_data],
    batch_size=16,
)
```

Most high-level inferencers also accept a plain list of `ImageData` or `BboxData` objects and create the appropriate generator internally.

## Inference

Model specs describe how to load a runtime. Inferencers wrap runtimes and return `ImageData` / `BboxData` results.

### Detection

```python
from cv_pipeliner import ImageData, YOLOv8_ModelSpec

images_data = [ImageData(image_path="images/example.jpg")]

model_spec = YOLOv8_ModelSpec(model_name="yolov8n.pt")
detection_inferencer = model_spec.load_detection_inferencer()

pred_images_data = detection_inferencer.predict(
    images_data_gen=images_data,
    score_threshold=0.25,
    batch_size_default=8,
    disable_tqdm=True,
)
```

`YOLOv8_ModelSpec` can load either a standard Ultralytics model name or a local/cloud `model_path` supported by `fsspec`.

### Classification

Classification inferencers can classify whole images or object crops. For crop classification, pass nested lists of `BboxData` or a `BatchGeneratorBboxData`.

```python
from cv_pipeliner import BatchGeneratorBboxData

bboxes_data_gen = BatchGeneratorBboxData(
    data=[image_data.bboxes_data],
    batch_size=16,
)

pred_bboxes_data = classification_inferencer.predict(
    data_gen=bboxes_data_gen,
    top_n=3,
    disable_tqdm=True,
)
```

The concrete `classification_inferencer` depends on the model backend, for example a TensorFlow classification model spec.

### Detection + Classification Pipeline

Use `PipelineModelSpec` when a detector finds objects and a classifier assigns final object labels.

```python
from cv_pipeliner import PipelineModelSpec

pipeline_spec = PipelineModelSpec(
    detection_model_spec=detection_model_spec,
    classification_model_spec=classification_model_spec,
)

pipeline_inferencer = pipeline_spec.load()
pred_images_data = pipeline_inferencer.predict(
    images_data_gen=images_data,
    detection_score_threshold=0.25,
    classification_top_n=1,
    disable_tqdm=True,
)
```

`classification_model_spec` is optional. If it is omitted, the pipeline behaves like detection postprocessing.

## Metrics

Metrics functions return pandas DataFrames.

```python
from cv_pipeliner import (
    get_df_classification_metrics,
    get_df_detection_metrics,
    get_df_pipeline_metrics,
)

df_detection = get_df_detection_metrics(
    true_images_data=true_images_data,
    pred_images_data=pred_images_data,
    minimum_iou=0.5,
)

df_pipeline = get_df_pipeline_metrics(
    true_images_data=true_images_data,
    pred_images_data=pred_images_data,
    minimum_iou=0.5,
)

df_classification = get_df_classification_metrics(
    n_true_bboxes_data=[image_data.bboxes_data for image_data in true_images_data],
    n_pred_bboxes_data=[image_data.bboxes_data for image_data in pred_images_data],
    pseudo_class_names=[],
)
```

Detection metrics include precision, recall, F1, IoU mean, and optional COCO metrics when the TensorFlow Object Detection API is installed and raw predictions are provided.

## Visualization

```python
from cv_pipeliner import ImageDataMatching
from cv_pipeliner import visualize_image_data, visualize_image_data_matching_side_by_side

image = visualize_image_data(image_data)

matching = ImageDataMatching(
    true_image_data=true_images_data[0],
    pred_image_data=pred_images_data[0],
    minimum_iou=0.5,
)
comparison = visualize_image_data_matching_side_by_side(
    image_data_matching=matching,
    error_type="detection",
)
```

Visualization helpers return NumPy arrays that can be saved with OpenCV/PIL, displayed in notebooks, or combined with other image utilities.

## Development

Install development dependencies:

```bash
poetry install
```

Run tests:

```bash
poetry run pytest
```

Run the executable documentation notebook test when optional notebook and model dependencies are installed:

```bash
poetry install --extras tensorflow --extras torch
poetry run pip install nbclient nbformat ipykernel
poetry run pytest tests/test_docs_notebooks.py
```

The notebook test executes [`docs/getting_started.ipynb`](docs/getting_started.ipynb), so it may download model weights on first use and can take longer than the unit test suite.
