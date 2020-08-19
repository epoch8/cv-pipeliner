import abc
from typing import List, Tuple

import numpy as np

from two_stage_pipeliner.core.inference_model import InferenceModel

Bbox = Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
Score = float

ImgBboxes = List[np.ndarray]
Bboxes = List[Bbox]
Scores = List[Score]

DetectionInput = List[np.ndarray]
DetectionOutput = List[
    Tuple[
        List[ImgBboxes],
        List[Bboxes],
        List[Scores]
    ]
]


class DetectionModel(InferenceModel):
    @abc.abstractmethod
    def load(self, checkpoint):
        InferenceModel.load(self, checkpoint)

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
