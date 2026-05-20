import abc
from typing import List, Type, Union

import numpy as np
from tqdm import tqdm

from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inferencers.base import Inferencer, ModelSpec, Runtime

EmbedderInput = List[np.ndarray]
EmbedderOutput = List[np.ndarray]


class EmbedderModelSpec(ModelSpec):
    @property
    @abc.abstractmethod
    def runtime_cls(self) -> Type["EmbedderRuntime"]:
        pass

    def load_embedder_inferencer(self) -> "EmbedderInferencer":
        return EmbedderInferencer(self.load_runtime())


class EmbedderRuntime(Runtime):
    def __init__(self, model_spec: EmbedderModelSpec):
        assert isinstance(model_spec, EmbedderModelSpec)
        super().__init__(model_spec=model_spec)

    @abc.abstractmethod
    def predict(self, input: EmbedderInput) -> EmbedderOutput:
        pass


class EmbedderInferencer(Inferencer):
    def __init__(self, runtime: EmbedderRuntime):
        assert isinstance(runtime, EmbedderRuntime)
        super().__init__(runtime)

    def predict(
        self,
        images_data_gen: Union[ImageData, BatchGeneratorImageData, "DataLoader"],
        batch_size_default: int = 16,
        disable_tqdm: bool = False,
    ) -> List[np.ndarray]:
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            DataLoader = None
        use_dataloader_torch = DataLoader is not None and isinstance(images_data_gen, DataLoader)

        if use_dataloader_torch:
            pass
        else:
            if isinstance(images_data_gen, list):
                images_data_gen = BatchGeneratorImageData(images_data_gen, batch_size=batch_size_default)
            assert isinstance(images_data_gen, BatchGeneratorImageData)

        pred_embeddings = []
        for images in tqdm(images_data_gen, total=len(images_data_gen), disable=disable_tqdm):
            if use_dataloader_torch:
                images = images[0]
            else:
                images = [image_data.image for image_data in images]

            embedding = self.model.predict(input=images)
            pred_embeddings.extend(embedding)
        return pred_embeddings

EmbedderRuntime = EmbedderRuntime
