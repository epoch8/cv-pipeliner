# 0.14.0
- Added new fields for `ImageData`:     `classification_score`, `top_n`, `labels_top_n`, `classification_scores_top_n` (similar to those fields in `BboxData`). They are used when `ImageData` is being submitted in `ClassificationInferencer` (thanks to @pixml27)
- Fix bug when `progress_callback` in `ClassificationInferencer` didn't update progress when tqdm is turned off.

# 0.13.1
- Added ModelSpec `MMPose_KeypointsRegressorModel` (thanks to @zakutnyaya)

# 0.13.0
- Added `meta_width` and `meta_height` for `ImageData` and `BboxData` to cache information about the height and length of source images in calculations. They also written and parsed in json() when image_data are opened once or when `force_include_meta=True` (new argument).
- Transformations with image_data `rotate_image_data`, `resize_image_data`, `thumbnail_image_data`, `crop_image_data`, `apply_perspective_transform_to_image_data` and `split_image_by_grid` now works with image_datas without source images like `image_data.image_path` and `image_data.image`, but only when `meta_width` and `meta_height` are available. This can be used for recalculating the changed annotation of the image о the image itself does not change. 
- `cv_pipeliner.utils.images_data.split_image_data_by_grid` is fixed and now saves additional bboxes_data, added required argument `remove_bad_coords: bool` for deleting bboxes that are not inside crops.
- In `cv_pipeliner.utils.datapipe` added new table stores: `ImageDataTableStoreDB` for store `ImageData`  inside any DataBase (SQLite or Postgress) and `ConnectedImageDataTableStore` for combining images from `TableStoreFiledir` and their corresponding `ImageData` with image_paths
- Added property `keypoints_n` for `ImageData` and `BboxData`.
- Added `convert_image_data_to_annotation` and `convert_annotation_to_image_data` in `cv_pipeliner.utils.label_studio` for annotations with one `RectangleLabels` and additional `KeyPointLabels` which takes into account the relationship between boxes and their corresponding keypoints.
- Updated `FifyOneSession` in `cv_pipeliner.utils.fiftyone`, now it supports cloud running.
- Added `FiftyOneImagesDataTableStore` in `cv_pipeliner.utils.datapipe` that converts `ImageData` to fiftyone database and vice versa.
- `get_image_size()` in `cv_pipeliner.utils.imagesize` now takes into account Exif tags
- DataConverter's `get_images_data_from_annots` now runs in parallel as possible.
- Added new functions `cv_pipeliner.utils.images.tf_resize_with_pad` and `rescale_bboxes_with_pad`.
- Added new field `use_default_preprocces_and_postprocess_input` in `YOLOv5_TFLite_ModelSpec` for using standard YOLOv5's preprocessing and postprocessing functions from previous paragraph

# 0.12.0
- Fix `bbox_data.json()` when values are written as np.int64
- Add `utils.datapipe.COCOLabelsFile` for datapipe
- Add argument `score_threshold` in `utils.images_datas.non_max_suppression_image_data_using_tf`
- Fixs in `parse_rectangle_labels_to_bbox_data` and `convert_image_data_to_rectangle_labels` in `utils.label_studio`
- Add new argument `thumbnail_size` in `cv_pipeliner.visualize_image_data`
- `ImageData.from_json` and `BboxData.from_json` now have arguments `image_data_cls` and `bbox_data_cls` for parsing JSON of child's classes.
- Add `threshold_score` in `utils.images_data.non_max_suppression_image_data_using_tf`
- `thumbnail_image` and `thumbnail_image_data` accepts also `int`, meaning size of `(int, int)`
- `YOLOv5_TFLite_ModelSpec` is now more stable
- Added `bbox_data.area` for calculating bbox's area
- The function `cv_pipeliner.utils.images_data.non_max_suppression_image_data` is refactored and works faster


# 0.11.1
- Add argument `warp_flags` in `cv_pipeliner.utils.images_data.rotate_image_data`
- Add cliping when `cv_pipeliner.utils.images.denormalize_bboxes`

# 0.11.0
- Added new inference models: `KeypointsRegressorModel`, `YOLOv5_DetectionModel` and `Tensorflow_EmbedderModel`
- Name of detectron2's models `Pytorch_DetectionModel` changed to `Detectron2_DetectionModel`
- `thumbnail_image_data` now works when image size need to be increased while aspecting ratio.
- `rotate_image_data` have 2 more arguments: `border_mode` and `border_value`
- Fix bug when `ImageData.from_json` doesn't give keypoints
- (FIXME) Added module `cv_pipeliner.utils.export` to export main models (Object Detection API, YOLOv5) to fixed `saved_model` and `.tflite` for mobile developments.
- Added better `__repr__` for `ImageData` and `BboxData` (removed `image` and `cropped_image` from them)


# 0.10.0
- `BboxData` now accepts floats xmin, ymin, xmax, ymax
- `ImageData.from_json()` can now be used on paths to json files
- Added `COCODataConverter` for COCO annotations
- Added tests for `DataConverter`
- Added `utils.imagesize` (taken from https://github.com/shibukawa/imagesize_py), with support of fsspec file-like objects
- Added DetectionModel for YOLOv5 in `inference_models.detection.yolov5`
- Pipeline's model logging level changed from INFO to DEBUG.
- Add `FiftyOneSession` for simplier using with FiftyOne (in `utils.fiftyone`)
- Added function `utils.images_data.flatten_additional_bboxes_data_in_image_data`
- `DetectionInferencer`, `ClassificationInferencer` and `PipelineInferencer` now can accept list of `ImageData` (of `BboxData` for `ClassificationInferencer`), with `batch_size_default=16` for Detection/Pipeline and `batch_size_default=32` for Classification
- Add method `.load_detection_inferencer()` to class `DetectionModelSpec` that loads model and returns corresponding `cv_pipeliner.inferencers.DetectionInferencer`
- Add method `.load_classification_inferencer()` to class `ClassificationModelSpec` that loads model and returns corresponding `cv_pipeliner.inferencers.ClassificationInferencer`
- Add method `.load_pipeline_inferencer()` to class `PipelineModelSpec` that loads model and returns corresponding `cv_pipeliner.inferencers.PipelineInferencer`
- Argument `classification_model_spec` is now `None` by default for `PipelineModelSpec`
- Add argument `return_as_pil_image: bool` set default as `False` in `visualize_image_data`, `visualize_images_data_side_by_side` and `visualize_image_data_matching_side_by_side`, that returns images as `PIL.Image.Image`


# 0.9.0
- Added `preprocess_input` and `input_size` for Object Detection API detectors;
- Fix exception bug in `non_max_suppression_image_data_using_tf` when `image_data.bboxes_data` are empty
- Added argument `additional_bboxes_data_depth` to `visualize_image_data` to display only first `depth` additional_bboxes_data
- Add variable `__version__` to package 

# 0.8.2
- Change function `cv_pipeliner.utils.images_datas.non_max_suppression_image_data_iou` -> `cv_pipeliner.utils.images_datas.non_max_suppression_image_data`
- Fix bugs in `non_max_suppression_image_data` when detection_score is None, additional_info is not updated

# 0.8.0-0.8.1

- Fix bugs

# 0.7.8

- Add changes in KFServing inferences.

# 0.7.6

- Add callbacks arguments in Inferencers for progress bars

# 0.7.5

- Add parallel inference for object detection (need joblib)
- Classification Inferencer now accept ImageData generators

# 0.7.0+

- Apps moved to `epoch8/cv-demostand`
- Tensorflow is not required
- Reporter for detection/pipeline works faster

# 0.6.2 (2021-04-14)

- Improved algorithm for matching true and predicted images data (ImageDataMatching) 

# 0.6.0-0.6.1 (2021-04-01)

- Add Mean Expected Steps metrics
- Major apps code refactor (migrate from Streamlit to Dash)

# 0.5.1 (2021-03-12)

- Add presicion@K classification metrics
- Fix `cv_pipeliner.complex_pipelines` module bug

# 0.5.0 (2021-03-11)

- Detector now can be pipeline itself
- Add complex pipeline: TextDetection

# 0.4.5 (2021-01-18)

- Package `object_detection` is now not required

# 0.4.4 (2021-01-18)

- Annotation in dataset browser is made better: top-n is now showed in annotation list.
- Backups in annotation mode are now made every hour.

# 0.4.3 (2021-01-13)

- Added quick annotation mode in the main app for model quality research
