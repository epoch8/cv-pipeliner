import pydantic
import abc
from dataclasses import dataclass
import importlib
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type, Union

import fsspec
import numpy as np


class ModelSpec(pydantic.BaseModel):
    id: str = None

    @abc.abstractproperty
    def inference_model_cls(self) -> Type['InferenceModel']:
        pass

    def load(self, **kwargs) -> Union['InferenceModel', 'Inferencer']:
        inference_model = self.inference_model_cls.get_loaded_model_by_id(self.id)
        if inference_model is None:
            inference_model = self.inference_model_cls(
                model_spec=self,
                **kwargs
            )
        return inference_model


class InferenceModel(abc.ABC):
    """
    Low-level class for models.
    To define the model, we need to create an object, then load checkpoint from given model_spec.

    Example:
        model_spec = ModelSpec(...)
        inference_model = InferenceModel(model_spec)
        input = inference_model.preprocess_input(input)
        output = inference_model.predict(input)

    2nd way:
        model_spec = ModelSpec(...)
        inference_model = model_spec.load()
        input = inference_model.preprocess_input(input)
        output = inference_model.predict(input)


    "input" and "output" types should be defined in the inheritance of this class.

    """
    _loaded_models: List[ModelSpec] = []

    @staticmethod
    def get_loaded_model_by_id(id: Optional[str]) -> Optional['InferenceModel']:
        if id is None:
            return None
        ids = [model._model_spec.id for model in InferenceModel._loaded_models]
        if id in ids:
            return InferenceModel._loaded_models[ids.index(id)]
        return None

    def __init__(self, model_spec: ModelSpec):
        self._model_spec = model_spec
        if self._model_spec.id is not None:
            InferenceModel._loaded_models.append(self)
        pass

    def __del__(self):
        ids = [model.id for model in InferenceModel._loaded_models]
        if self.model_spec.id is not None and self.model_spec.id in ids:
            InferenceModel._loaded_models.pop(ids.index(self.model_spec.id))

    @abc.abstractmethod
    def predict(self, input):
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> Tuple[int, int]:
        pass

    @property
    def model_spec(self):
        return self._model_spec


def get_preprocess_input_from_script_file(
    script_file: Union[str, Path]
) -> Callable[[List[np.ndarray]], np.ndarray]:
    with fsspec.open(script_file, 'r') as src:
        script_code = src.read()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        module_folder = tmpdirname / 'module'
        module_folder.mkdir()
        script_file = module_folder / f'preprocess_input_{tmpdirname.name}.py'
        with open(script_file, 'w') as out:
            out.write(script_code)
        sys.path.append(str(script_file.parent.absolute()))
        module = importlib.import_module(script_file.stem)
        importlib.reload(module)
        sys.path.pop()
    return module.preprocess_input
