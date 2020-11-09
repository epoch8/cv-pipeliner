from dataclasses import dataclass

from cv_pipeliner.inference_models.detection.core import DetectionModelSpec
from cv_pipeliner.inference_models.detection.object_detection_api import (
    ObjectDetectionAPI_ModelSpec,
    ObjectDetectionAPI_pb_ModelSpec,
    ObjectDetectionAPI_TFLite_ModelSpec
)
from cv_pipeliner.inference_models.classification.core import ClassificationModelSpec
from cv_pipeliner.inference_models.classification.tensorflow import TensorFlow_ClassificationModelSpec

from typing import Union


@dataclass
class DetectionModelDefinition:
    description: str
    model_spec: Union[
        DetectionModelSpec, ObjectDetectionAPI_ModelSpec,
        ObjectDetectionAPI_pb_ModelSpec, ObjectDetectionAPI_TFLite_ModelSpec
    ]
    score_threshold: float
    model_index: str


@dataclass
class ClassificationDefinition:
    description: str
    model_spec: Union[ClassificationModelSpec, TensorFlow_ClassificationModelSpec]  # noqa: E501
    model_index: str
