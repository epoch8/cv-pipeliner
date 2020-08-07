from typing import List, Tuple
import numpy as np

from two_stage_pipeliner.core.inference_model import InferenceModel

ImageInput = np.ndarray
Label = str
Score = float

Labels = List[Label]
Scores = List[Score]

ClassificationInput = List[List[ImageInput]]
ClassificationOutput = Tuple[List[Labels], List[Scores]]


class ClassificationModel(InferenceModel):
    def __init__(self):
        super(ClassificationModel, self).__init__()

    def predict(self, input: ClassificationInput) -> ClassificationOutput:
        pass
