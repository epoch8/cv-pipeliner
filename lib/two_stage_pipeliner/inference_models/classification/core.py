import abc
from typing import List, Tuple
import numpy as np

from two_stage_pipeliner.core.inference_model import InferenceModel

Label = str
Score = float

Labels = List[Label]
Scores = List[Score]

ClassificationInput = List[List[np.ndarray]]
ClassificationOutput = Tuple[List[Labels], List[Scores]]


class ClassificationModel(InferenceModel):
    @abc.abstractmethod
    def load(self, checkpoint):
        InferenceModel.load(self, checkpoint)

    @abc.abstractmethod
    def predict(self, input: ClassificationInput) -> ClassificationOutput:
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> int:
        pass

    @abc.abstractproperty
    def class_names(self) -> List[str]:
        pass
