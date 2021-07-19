import numpy as np
from typing import List
from tqdm import tqdm
from cv_pipeliner.inference_models.embedder.core import EmbedderModel
from cv_pipeliner.core.inferencer import Inferencer
from torch.utils.data import DataLoader


class EmbedderInferencer(Inferencer):
    def __init__(self, model: EmbedderModel):
        assert isinstance(model, EmbedderModel)
        super().__init__(model)

    def predict(
        self,
        loader: DataLoader
    ) -> List[np.ndarray]:

        pred_embeddings = []
        for images, _ in tqdm(loader, total=len(loader)):

            embedding = self.model.predict(input=images)
            pred_embeddings.extend(embedding)
        return pred_embeddings
