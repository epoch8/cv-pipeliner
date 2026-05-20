import abc
from importlib.metadata import version
from typing import Any, List, Optional, Type, Union

from packaging.version import Version

if Version(version("pydantic")) < Version("2.0.0"):
    from pydantic import BaseModel as PydanticBaseModel
else:
    from pydantic.v1 import BaseModel as PydanticBaseModel

from cv_pipeliner.core.batch_generator import BatchGenerator


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class ModelSpec(BaseModel):
    id: str = None

    @property
    @abc.abstractmethod
    def runtime_cls(self) -> Type["Runtime"]:
        pass

    def load_runtime(self, **kwargs) -> "Runtime":
        runtime = Runtime.get_loaded_runtime_by_id(self.id)
        if runtime is None:
            runtime = self.runtime_cls(model_spec=self, **kwargs)
        return runtime

    def load(self, **kwargs) -> "Runtime":
        return self.load_runtime(**kwargs)


class Runtime(abc.ABC):
    _loaded_runtimes: List["Runtime"] = []

    @staticmethod
    def get_loaded_runtime_by_id(id: Optional[str]) -> Optional["Runtime"]:
        if id is None:
            return None
        ids = [runtime.spec.id for runtime in Runtime._loaded_runtimes]
        if id in ids:
            return Runtime._loaded_runtimes[ids.index(id)]
        return None

    def __init__(self, model_spec: ModelSpec = None):
        self._spec = model_spec
        if self._spec is not None and self._spec.id is not None:
            Runtime._loaded_runtimes.append(self)

    def __del__(self):
        if getattr(self, "_spec", None) is None or self._spec.id is None:
            return
        ids = [runtime.spec.id for runtime in Runtime._loaded_runtimes]
        if self._spec.id in ids:
            Runtime._loaded_runtimes.pop(ids.index(self._spec.id))

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
