import abc


class InferenceModel(abc.ABC):
    @abc.abstractmethod
    def load(self, checkpoint):
        self.checkpoint = checkpoint
        pass

    @abc.abstractmethod
    def predict(self, input):
        pass

    @abc.abstractmethod
    def preprocess_input(self, input):
        pass

    @abc.abstractproperty
    def input_size(self) -> int:
        pass
