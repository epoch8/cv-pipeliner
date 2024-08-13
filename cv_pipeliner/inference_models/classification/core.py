import abc
from typing import List, Tuple, Type

import numpy as np

from cv_pipeliner.core.inference_model import InferenceModel, ModelSpec

Label = str
Score = float

Labels_Top_N = List[List[Label]]
Scores_Top_N = List[List[Score]]

ClassificationInput = List[np.ndarray]
ClassificationOutput = Tuple[Labels_Top_N, Scores_Top_N]


class ClassificationModelSpec(ModelSpec):
    @abc.abstractproperty
    def inference_model_cls(self) -> Type["ClassificationModel"]:
        pass

    def load_classification_inferencer(self) -> "ClassificationInferencer":
        from cv_pipeliner.inferencers.classification import ClassificationInferencer

        return ClassificationInferencer(self.load())


class ClassificationModel(InferenceModel):
    def __init__(self, model_spec: ClassificationModelSpec):
        assert isinstance(model_spec, ClassificationModelSpec)
        super().__init__(model_spec)

    @abc.abstractmethod
    def predict(self, input: ClassificationInput, top_n: int = 1) -> ClassificationOutput:
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
