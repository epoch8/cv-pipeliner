import abc
from two_stage_pipeliner.core.batch_generator import BatchGenerator
from two_stage_pipeliner.core.inference_model import InferenceModel


class Inferencer(abc.ABC):
    """
    High-level class for inference.

    The class accepts loaded InferenceModel and make predictions
    by using high-level objects (BatchGenerators).

    Example:
        model_spec = ModelSpec(...)
        inference_model = model_spec.load_model()
        inferencer = Inferencer(inference_model)
        data_gen = BatchGenerator(data, batch_size=16)
        pred_data = inferencer.predict(data_gen)

    """

    def __init__(self, model: InferenceModel):
        self.model = model

    @abc.abstractmethod
    def predict(self, batch_generator: BatchGenerator):
        pass
