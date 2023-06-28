import abc
from pathlib import Path
from typing import Union, List

from cv_pipeliner.core.inference_model import ModelSpec


class Reporter(abc.ABC):
    @abc.abstractmethod
    def report(self, models_specs: List[ModelSpec], tags: List[str], output_directory: Union[str, Path]):
        pass
