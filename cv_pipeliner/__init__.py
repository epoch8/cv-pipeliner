from importlib import import_module

__version__ = "0.22.0"

_LAZY_IMPORTS = {
    "BatchGeneratorBboxData": "cv_pipeliner.batch_generators.bbox_data",
    "BatchGeneratorImageData": "cv_pipeliner.batch_generators.image_data",
    "BboxData": "cv_pipeliner.core.data",
    "ImageData": "cv_pipeliner.core.data",
    "COCODataConverter": "cv_pipeliner.data_converters.coco",
    "JSONDataConverter": "cv_pipeliner.data_converters.json",
    "SuperviselyDataConverter": "cv_pipeliner.data_converters.supervisely",
    "YOLODataConverter": "cv_pipeliner.data_converters.yolo",
    "YOLOMasksDataConverter": "cv_pipeliner.data_converters.yolo",
    "TensorFlow_ClassificationModelSpec": "cv_pipeliner.inferencers.classification.tensorflow",
    "TensorFlow_ClassificationModelSpec_TFServing": "cv_pipeliner.inferencers.classification.tensorflow",
    "YOLOv5_ModelSpec": "cv_pipeliner.inferencers.detection.yolov5",
    "YOLOv5_TFLite_ModelSpec": "cv_pipeliner.inferencers.detection.yolov5",
    "YOLOv5_TFLiteWithNMS_ModelSpec": "cv_pipeliner.inferencers.detection.yolov5",
    "YOLOv8_ModelSpec": "cv_pipeliner.inferencers.detection.yolov8",
    "PyTorch_EmbedderModelSpec": "cv_pipeliner.inferencers.embedder.pytorch",
    "TensorFlow_EmbedderModelSpec": "cv_pipeliner.inferencers.embedder.tensorflow",
    "MMPose_KeypointsRegressorModelSpec_TFLite": "cv_pipeliner.inferencers.keypoints_regressor.mmpose",
    "TensorFlow_KeypointsRegressorModelSpec": "cv_pipeliner.inferencers.keypoints_regressor.tensorflow",
    "TensorFlow_KeypointsRegressorModelSpec_TFServing": "cv_pipeliner.inferencers.keypoints_regressor.tensorflow",
    "PipelineModelSpec": "cv_pipeliner.inferencers.pipeline",
    "ClassificationInferencer": "cv_pipeliner.inferencers.classification",
    "DetectionInferencer": "cv_pipeliner.inferencers.detection",
    "KeypointsRegressorInferencer": "cv_pipeliner.inferencers.keypoints_regressor",
    "PipelineInferencer": "cv_pipeliner.inferencers.pipeline",
    "get_df_classification_metrics": "cv_pipeliner.metrics.classification",
    "get_df_detection_metrics": "cv_pipeliner.metrics.detection",
    "BboxDataMatching": "cv_pipeliner.metrics.image_data_matching",
    "ImageDataMatching": "cv_pipeliner.metrics.image_data_matching",
    "intersection_over_union": "cv_pipeliner.metrics.image_data_matching",
    "pairwise_intersection_over_union": "cv_pipeliner.metrics.image_data_matching",
    "get_df_pipeline_metrics": "cv_pipeliner.metrics.pipeline",
    "FiftyOneSession": "cv_pipeliner.utils.fiftyone",
    "FifyOneSession": "cv_pipeliner.utils.fiftyone",
    "concat_images": "cv_pipeliner.utils.images",
    "draw_rectangle": "cv_pipeliner.utils.images",
    "put_text_on_image": "cv_pipeliner.utils.images",
    "thumbnail_image": "cv_pipeliner.utils.images",
    "apply_perspective_transform_to_image_data": "cv_pipeliner.utils.images_datas",
    "combine_mask_polygons_to_one_polygon": "cv_pipeliner.utils.images_datas",
    "crop_image_data": "cv_pipeliner.utils.images_datas",
    "flatten_additional_bboxes_data_in_image_data": "cv_pipeliner.utils.images_datas",
    "get_perspective_matrix_for_base_keypoints": "cv_pipeliner.utils.images_datas",
    "non_max_suppression_image_data": "cv_pipeliner.utils.images_datas",
    "non_max_suppression_image_data_using_tf": "cv_pipeliner.utils.images_datas",
    "resize_image_data": "cv_pipeliner.utils.images_datas",
    "rotate_image_data": "cv_pipeliner.utils.images_datas",
    "split_image_by_grid": "cv_pipeliner.utils.images_datas",
    "thumbnail_image_data": "cv_pipeliner.utils.images_datas",
    "uncrop_bboxes_data": "cv_pipeliner.utils.images_datas",
    "get_image_size": "cv_pipeliner.utils.imagesize",
    "visualize_image_data": "cv_pipeliner.visualizers.core.image_data",
    "visualize_image_data_matching_side_by_side": "cv_pipeliner.visualizers.core.image_data_matching",
}

__all__ = [*_LAZY_IMPORTS, "__version__"]


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(_LAZY_IMPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *_LAZY_IMPORTS])
