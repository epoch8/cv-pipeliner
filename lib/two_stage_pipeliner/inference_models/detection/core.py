import abc
from typing import List, Tuple, ClassVar

import numpy as np

from two_stage_pipeliner.core.inference_model import ModelSpec, InferenceModel

Bbox = Tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)
Score = float

CroppedImages = List[np.ndarray]
Bboxes = List[Bbox]
Scores = List[Score]

DetectionInput = List[np.ndarray]
DetectionOutput = Tuple[
    List[CroppedImages],
    List[Bboxes],
    List[Scores]
]


class DetectionModelSpec(ModelSpec):

    @abc.abstractproperty
    def inference_model(self) -> ClassVar['DetectionModel']:
        pass


class DetectionModel(InferenceModel):
    @abc.abstractmethod
    def load(self, model_spec: DetectionModelSpec):
        assert isinstance(model_spec, DetectionModelSpec)
        super().load(model_spec)

    @abc.abstractmethod
    def predict(self, input: DetectionInput,
                score_threshold: float) -> DetectionOutput:
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> int:
        pass
