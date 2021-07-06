import numpy as np
from typing import List
from tqdm import tqdm
from cv_pipeliner.inference_models.embedder.core import EmbedderModel
from cv_pipeliner.core.inferencer import Inferencer
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData


class EmbedderInferencer(Inferencer):
    def __init__(self, model: EmbedderModel):
        assert isinstance(model, EmbedderModel)
        super().__init__(model)

    def predict(
        self,
        image_data_gen: BatchGeneratorImageData,
        disable_tqdm: bool = False
    ) -> List[np.ndarray]:

        pred_embeddings = []
        with tqdm(total=len(image_data_gen.data),
                  disable=disable_tqdm) as pbar:
            for image_data in image_data_gen:
                input = image_data
                embedding = self.model.predict(
                    input=input,
                )
                pred_embeddings.extend(embedding)
                pbar.update(len(image_data))

        return pred_embeddings
