import abc
from typing import List, Tuple, ClassVar
import numpy as np

from two_stage_pipeliner.core.inference_model import ModelSpec, InferenceModel

Label = str
Score = float

Labels = List[Label]
Scores = List[Score]

ClassificationInput = List[List[np.ndarray]]
ClassificationOutput = Tuple[List[Labels], List[Scores]]


class ClassificationModelSpec(ModelSpec):

    @abc.abstractproperty
    def inference_model(self) -> ClassVar['ClassificationModel']:
        pass


class ClassificationModel(InferenceModel):
    @abc.abstractmethod
    def load(self, model_spec: ClassificationModelSpec):
        assert isinstance(model_spec, ClassificationModelSpec)
        super().load(model_spec)

    @abc.abstractmethod
    def predict(self, input: ClassificationInput) -> ClassificationOutput:
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> Tuple[int, int]:
        pass

    @abc.abstractproperty
    def class_names(self) -> List[str]:
        pass
