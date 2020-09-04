import abc
from pathlib import Path
from typing import Union

from two_stage_pipeliner.core.inference_model import ModelSpec
from two_stage_pipeliner.core.inferencer import Inferencer


class Reporter(abc.ABC):
    @abc.abstractmethod
    def report(
        self,
        model_spec: ModelSpec,
        output_directory: Union[str, Path],
    ):
        pass
