import abc
from typing import List, Type

import numpy as np

from cv_pipeliner.core.inference_model import InferenceModel, ModelSpec

EmbedderInput = List[np.ndarray]
EmbedderOutput = List[np.ndarray]


class EmbedderModelSpec(ModelSpec):
    @abc.abstractproperty
    def inference_model_cls(self) -> Type["EmbedderModel"]:
        pass

    def load_embedder_inferencer(self) -> "EmbedderInferencer":
        from cv_pipeliner.inferencers.embedder import EmbedderInferencer

        return EmbedderInferencer(self.load())


class EmbedderModel(InferenceModel):
    def __init__(self, model_spec: EmbedderModelSpec):
        assert isinstance(model_spec, EmbedderModelSpec)
        super().__init__(model_spec)

    @abc.abstractmethod
    def predict(
        self,
        input: EmbedderInput,
    ) -> EmbedderOutput:
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> int:
        pass
