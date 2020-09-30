import abc
from typing import Tuple, Type


class ModelSpec(abc.ABC):
    @abc.abstractproperty
    def inference_model_cls(self) -> Type['InferenceModel']:
        pass

    def __init__(self) -> 'InferenceModel':
        inference_model = self.inference_model_cls(self)
        return inference_model


class InferenceModel(abc.ABC):
    """
    Low-level class for models.
    To define the model, we need to create an object, then load checkpoint from given model_spec.

    Example:
        model_spec = ModelSpec(...)
        inference_model = InferenceModel()
        inference_model.load(model_spec)
        input = inference_model.preprocess_input(input)
        output = inference_model.predict(input)

    2nd way:
        model_spec = ModelSpec(...)
        inference_model = model_spec.load_model()
        input = inference_model.preprocess_input(input)
        output = inference_model.predict(input)


    "input" and "output" types should be defined in the inheritance of this class.

    """

    def __init__(self, model_spec: ModelSpec):
        self._model_spec = model_spec
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

    @property
    def model_spec(self):
        return self._model_spec
