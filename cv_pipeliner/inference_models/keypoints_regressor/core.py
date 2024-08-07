import abc
from typing import List, Tuple, Type

import numpy as np

from cv_pipeliner.core.inference_model import InferenceModel, ModelSpec

Keypoint = Tuple[int, int]
Keypoints = List[Keypoint]

KeypointsRegressorInput = List[np.ndarray]
KeypointsRegressorOutput = List[Keypoints]


class KeypointsRegressorModelSpec(ModelSpec):
    @abc.abstractproperty
    def inference_model_cls(self) -> Type["KeypointsRegressorModel"]:
        pass

    def load_keypoints_regressor_inferencer(self) -> "KeypointsRegressorInferencer":
        from cv_pipeliner.inferencers.keypoints_regressor import (
            KeypointsRegressorInferencer,
        )

        return KeypointsRegressorInferencer(self.load())


class KeypointsRegressorModel(InferenceModel):
    def __init__(self, model_spec: KeypointsRegressorModelSpec):
        assert isinstance(model_spec, KeypointsRegressorModelSpec)
        super().__init__(model_spec)

    @abc.abstractmethod
    def predict(
        self,
        input: KeypointsRegressorInput,
    ) -> KeypointsRegressorOutput:
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> Tuple[int, int]:
        pass
