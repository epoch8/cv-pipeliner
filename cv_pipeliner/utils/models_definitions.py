from dataclasses import dataclass

from cv_pipeliner.inference_models.detection.core import DetectionModelSpec
from cv_pipeliner.inference_models.detection.object_detection_api import (
    ObjectDetectionAPI_ModelSpec,
    ObjectDetectionAPI_pb_ModelSpec,
    ObjectDetectionAPI_TFLite_ModelSpec,
    ObjectDetectionAPI_KFServing
)
from cv_pipeliner.inference_models.classification.core import ClassificationModelSpec
from cv_pipeliner.inference_models.classification.tensorflow import (
    TensorFlow_ClassificationModelSpec, TensorFlow_ClassificationModelSpec_TFServing
)
from cv_pipeliner.inference_models.classification.dummy import Dummy_ClassificationModelSpec

from typing import Union


@dataclass
class DetectionModelDefinition:
    description: str
    model_spec: Union[
        DetectionModelSpec, ObjectDetectionAPI_ModelSpec,
        ObjectDetectionAPI_pb_ModelSpec, ObjectDetectionAPI_TFLite_ModelSpec,
        ObjectDetectionAPI_KFServing
    ]
    score_threshold: float
    model_index: str


@dataclass
class ClassificationModelDefinition:
    description: str
    model_spec: Union[
        ClassificationModelSpec,
        TensorFlow_ClassificationModelSpec,
        TensorFlow_ClassificationModelSpec_TFServing,
        Dummy_ClassificationModelSpec
    ]
    model_index: str


@dataclass
class PipelineModelDefinition:
    description: str
    detection_model_definition: DetectionModelDefinition
    classification_model_definition: ClassificationModelDefinition
