import abc
from two_stage_pipeliner.core.batch_generator import BatchGenerator
from two_stage_pipeliner.core.inferencer import Inferencer


class MetricsCounter(abc.ABC):
    def __init__(self, inferencer: Inferencer):
        self.inferencer = inferencer
        super(MetricsCounter, self).__init__()

    @abc.abstractmethod
    def score(self, batch_generator: BatchGenerator):
        pass
