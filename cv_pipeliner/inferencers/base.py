import abc
from typing import Any, Type, Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict

from cv_pipeliner.core.batch_generator import BatchGenerator


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelSpec(BaseModel):
    @property
    @abc.abstractmethod
    def runtime_cls(self) -> Type["Runtime"]:
        pass

    def load_runtime(self, **kwargs) -> "Runtime":
        return self.runtime_cls(model_spec=self, **kwargs)

    def load(self, **kwargs) -> "Runtime":
        return self.load_runtime(**kwargs)


class Runtime(abc.ABC):
    def __init__(self, model_spec: ModelSpec = None):
        self._spec = model_spec

    @property
    def spec(self):
        return self._spec

    @property
    def model_spec(self):
        return self._spec

    @abc.abstractmethod
    def predict(self, input: Any):
        pass

    @abc.abstractmethod
    def preprocess_input(self, input: Any):
        pass

    @property
    @abc.abstractmethod
    def input_size(self):
        pass


class Inferencer(abc.ABC):
    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.model = runtime

    @abc.abstractmethod
    def predict(self, data: Union[Any, BatchGenerator]):
        pass


RuntimeModel = Runtime

__all__ = ["BaseModel", "Inferencer", "ModelSpec", "Runtime", "RuntimeModel"]
