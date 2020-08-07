import abc


class InferenceModel(abc.ABC):
    def __init__(self):
        super(InferenceModel, self).__init__()

    @abc.abstractmethod
    def load(self, checkpoint):
        pass

    @abc.abstractmethod
    def predict(self, input):
        pass

    @abc.abstractclassmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> int:
        pass
