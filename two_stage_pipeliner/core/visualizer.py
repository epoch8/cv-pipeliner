import abc
from typing import List, Union

from two_stage_pipeliner.core.data import BboxData, ImageData
from two_stage_pipeliner.core.inferencer import Inferencer


class Visualizer(abc.ABC):
    def __init__(self, inferencer: Inferencer):
        self.inferencer = inferencer
        super(Visualizer, self).__init__()

    @abc.abstractmethod
    def visualize(self, input: List[Union[BboxData, ImageData]]):
        pass
