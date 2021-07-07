import torch
import numpy as np
from cv_pipeliner.core.inference_model import (
    get_preprocess_input_from_script_file
)
from cv_pipeliner.inference_models.embedder.core import (
    EmbedderModelSpec, EmbedderModel, EmbedderInput, EmbedderOutput
)
from pathy import Pathy
from dataclasses import dataclass
from typing import List, Tuple, Callable, Union, Type
from pathlib import Path


@dataclass
class PyTorch_EmbedderModelSpec(EmbedderModelSpec):
    model_path: Union[str, Pathy]
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str,
                            Path, None] = None

    @property
    def inference_model_cls(self) -> Type['PyTorch_EmbedderModel']:
        from cv_pipeliner.inference_models.embedder.pytorch import (
            PyTorch_EmbedderModel)
        return PyTorch_EmbedderModel


class PyTorch_EmbedderModel(EmbedderModel):
    def _load_pytorch_model_spec(
        self,
        model_spec: PyTorch_EmbedderModelSpec
    ):
        self.model = torch.jit.load(str(model_spec.model_path))
        self.model.eval()

    def __init__(
        self,
        model_spec: PyTorch_EmbedderModelSpec
    ):
        super().__init__(model_spec)

        self._load_pytorch_model_spec(model_spec)
        self._raw_predict = self._raw_predict_pytorch

        if isinstance(model_spec.preprocess_input, str) or isinstance(
                        model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(
                script_file=model_spec.preprocess_input
            )
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

    def _raw_predict_pytorch(
        self,
        images: np.ndarray
    ):
        outputs = []
        self.model.cuda()
        with torch.no_grad():
            images = images.cuda().float()
            prediction = self.model(images).cpu().detach().numpy()
            outputs.extend(prediction)
        return outputs

    def preprocess_input(self, input: EmbedderInput):
        return self._preprocess_input(input)

    def predict(
        self,
        input: EmbedderInput,
    ) -> EmbedderOutput:
        input = self.preprocess_input(input)
        predictions = self._raw_predict_pytorch(input)
        return predictions

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size
