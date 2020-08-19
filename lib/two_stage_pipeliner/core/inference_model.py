import abc
from typing import Tuple


class Checkpoint(abc.ABC):
    pass


class InferenceModel(abc.ABC):
    """Low-level class for models.
    To define the model, we need to create an object, then load checkpoint.

    Example:
    inference_model = InferenceModel()
    inference_model.load(checkpoint)
    output = inference_model.predict(inference_model.preprocess_input(input))

    "input" and "output" types should be defined in the inheritance of this class.

    """

    @abc.abstractmethod
    def load(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        pass

    @abc.abstractmethod
    def predict(self, input):
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> Tuple[int, int]:
        pass
