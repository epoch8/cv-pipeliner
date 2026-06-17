def test_detection_backend_imports_are_available_from_inferencers():
    from cv_pipeliner.inferencers.detection.yolov5 import YOLOv5Runtime, YOLOv5_ModelSpec

    assert YOLOv5Runtime is not None
    assert YOLOv5_ModelSpec is not None


def test_classification_backend_imports_are_available_from_inferencers():
    from cv_pipeliner.inferencers.classification.tensorflow import (
        TensorFlowClassificationRuntime,
        TensorFlow_ClassificationModelSpec,
    )

    assert TensorFlowClassificationRuntime is not None
    assert TensorFlow_ClassificationModelSpec is not None


def test_package_level_exports_stay_available():
    from cv_pipeliner import PipelineInferencer, PipelineModelSpec, TensorFlow_ClassificationModelSpec, YOLOv5_ModelSpec

    assert PipelineInferencer is not None
    assert PipelineModelSpec is not None
    assert TensorFlow_ClassificationModelSpec is not None
    assert YOLOv5_ModelSpec is not None


def test_new_task_core_imports_are_available():
    from cv_pipeliner.inferencers.classification.core import ClassificationRuntime, ClassificationModelSpec
    from cv_pipeliner.inferencers.detection.core import DetectionRuntime, DetectionModelSpec
    from cv_pipeliner.inferencers.embedder.core import EmbedderRuntime, EmbedderModelSpec
    from cv_pipeliner.inferencers.keypoints_regressor.core import KeypointsRegressorRuntime, KeypointsRegressorModelSpec
    from cv_pipeliner.inferencers.pipeline import PipelineModelSpec

    assert ClassificationRuntime is not None
    assert ClassificationModelSpec is not None
    assert DetectionRuntime is not None
    assert DetectionModelSpec is not None
    assert EmbedderRuntime is not None
    assert EmbedderModelSpec is not None
    assert KeypointsRegressorRuntime is not None
    assert KeypointsRegressorModelSpec is not None
    assert PipelineModelSpec is not None
