import abc
from typing import List, Tuple, Type

import numpy as np

from cv_pipeliner.core.inference_model import ModelSpec, InferenceModel

Bbox = Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
Score = float
Class = str

Bboxes = List[Bbox]
Scores = List[Score]
Classes = List[Class]
Keypoints = List[Tuple[int, int]]

DetectionInput = List[np.ndarray]
DetectionOutput = Tuple[
    List[Bboxes],
    List[Keypoints],
    List[Scores],
    List[Classes],  # Optional exit
    List[Scores]  # Optional exit
]


class DetectionModelSpec(ModelSpec):
    class_names: List[str] = None  # optional

    @abc.abstractproperty
    def inference_model_cls(self) -> Type['DetectionModel']:
        pass

    def load_detection_inferencer(self) -> 'DetectionInferencer':
        from cv_pipeliner.inferencers.detection import DetectionInferencer
        return DetectionInferencer(self.load())


class DetectionModel(InferenceModel):
    @abc.abstractmethod
    def __init__(self, model_spec: DetectionModelSpec):
        assert isinstance(model_spec, DetectionModelSpec)
        super().__init__(model_spec)

    @abc.abstractmethod
    def predict(
        self,
        input: DetectionInput,
        score_threshold: float,
        classification_top_n: int = None
    ) -> DetectionOutput:
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> int:
        pass
