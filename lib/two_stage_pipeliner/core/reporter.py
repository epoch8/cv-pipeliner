import abc
from pathlib import Path
from typing import Union

from two_stage_pipeliner.core.batch_generator import BatchGenerator
from two_stage_pipeliner.core.inferencer import Inferencer


class Reporter(abc.ABC):
    @abc.abstractmethod
    def report(self,
               inferencer: Inferencer,
               data_generator: BatchGenerator,
               directory: Union[str, Path]):
        pass
