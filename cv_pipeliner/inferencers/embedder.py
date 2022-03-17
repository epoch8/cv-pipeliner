import numpy as np

from typing import List, Union
from tqdm import tqdm

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inference_models.embedder.core import EmbedderModel
from cv_pipeliner.core.inferencer import Inferencer
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData


class EmbedderInferencer(Inferencer):
    def __init__(self, model: EmbedderModel):
        assert isinstance(model, EmbedderModel)
        super().__init__(model)

    def predict(
        self,
        images_data_gen: Union[ImageData, BatchGeneratorImageData, 'DataLoader'],
        batch_size_default: int = 16,
        disable_tqdm: bool = False
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
