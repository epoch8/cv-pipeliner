import abc
import numpy as np
from typing import List, Type
from torch.utils.data import DataLoader
from cv_pipeliner.core.inference_model import ModelSpec, InferenceModel

EmbedderInput = DataLoader
EmbedderOutput = List[np.ndarray]


class EmbedderModelSpec(ModelSpec):

    @abc.abstractproperty
    def inference_model_cls(self) -> Type['EmbedderModel']:
        pass


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
