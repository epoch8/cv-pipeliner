import abc
from two_stage_pipeliner.core.batch_generator import BatchGenerator
from two_stage_pipeliner.core.inference_model import InferenceModel


class Inferencer(abc.ABC):
    """High-level class for inference.

    The class accepts loaded InferenceModel and make predictions
    by using high-level objects (e.g., BboxData, ImageData)

    """

    def __init__(self, model: InferenceModel):
        self.model = model

    @abc.abstractmethod
    def predict(self, batch_generator: BatchGenerator):
        pass
