from typing import List, Tuple

import numpy as np

from two_stage_pipeliner.core.inference_model import InferenceModel

ImageInput = np.ndarray
Bbox = Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
Score = float

ImgBboxes = List[ImageInput]
Bboxes = List[Bbox]
Scores = List[Score]

DetectionInput = List[ImageInput]
DetectionOutput = List[
    Tuple[
        List[ImgBboxes],
        List[Bboxes],
        List[Scores]
    ]
]


class DetectionModel(InferenceModel):
    def __init__(self, score_threshold: float):
        super(DetectionModel, self).__init__()
        self.score_threshold = score_threshold

    def load(self, checkpoint):
        pass

    def predict(self, input: DetectionInput) -> DetectionOutput:
        return None

    def preprocess_input(self, input):
        pass

    @property
    def input_size(self) -> int:
        return None
