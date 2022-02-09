# 0.10.0
- `ImageData.from_json()` can now be used on paths to json files
- Added `COCODataConverter` for COCO annotations
- Added tests for `DataConverter`
- Added `utils.imagesize` (taken from https://github.com/shibukawa/imagesize_py), with support of fsspec file-like objects
- Added DetectionModel for YOLOv5 in `inference_models.detection.yolov5`
- Pipeline's model logging level changed from INFO to DEBUG.

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
