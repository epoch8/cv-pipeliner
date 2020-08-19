from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from tensorflow.keras.models import load_model
from tensorflow.python.keras.engine.training import Model as TfModelType

from two_stage_pipeliner.inference_models.classification.core import ClassificationModel, \
    ClassificationInput, ClassificationOutput
from two_stage_pipeliner.inference_models.classification.tf.specs import ClassifierModelSpecTF


class ClassifierTF(ClassificationModel):
    def load(self, checkpoint: ClassifierModelSpecTF):
        ClassificationModel.load(self, checkpoint)
        model_spec = checkpoint
        if isinstance(model_spec.model_path, str) or isinstance(
            model_spec.model_path, Path
        ):
            self.model = load_model(str(model_spec.model_path))
        elif isinstance(model_spec.model_path, TfModelType):
            self.model = model_spec.model_path
        self.model_spec = model_spec
        assert model_spec.num_classes == int(self.model.output.shape[-1])
        self.num_classes = model_spec.num_classes
        self._class_names = model_spec.class_names
        assert self.num_classes == len(self.class_names)
        self.id_to_class_name = {
            id: class_name for id, class_name in enumerate(self.class_names)
        }
        self.disable_tqdm = False
        self.batch_size = 16

    def _split_chunks(self,
                      l: np.ndarray,
                      chunk_sizes: List[int]) -> List[np.ndarray]:
        cnt = 0
        chunks = []
        for chunk_size in chunk_sizes:
            chunks.append(l[cnt: cnt + chunk_size])
            cnt += chunk_size
        return chunks

    def _raw_predict(self,
                     images: List[List[np.ndarray]]) -> np.ndarray:

        predictions = []

        shapes = [len(sample_batch) for sample_batch in images]
        images = np.array(
            [item for sublist in images for item in sublist]
        )

        with tqdm(total=len(images), disable=self.disable_tqdm) as pbar:
            for i in range(0, len(images), self.batch_size):
                batch = images[i: i + self.batch_size]

                predictions_batch = self.model.predict(batch)
                predictions.append(predictions_batch)

                pbar.update(len(batch))

                del batch

        predictions = np.concatenate(predictions)
        predictions = self._split_chunks(predictions, shapes)

        return predictions

    def _postprocess_predictions(self,
                                 predictions: np.ndarray) -> ClassificationOutput:

        max_scores_idxs = [np.argmax(pred, axis=1) for pred in predictions]
        pred_scores = [np.max(pred, axis=1) for pred in predictions]

        pred_labels = [
            [self.id_to_class_name[i] for i in pred_class]
            for pred_class in max_scores_idxs
        ]

        return pred_labels, pred_scores

    def predict(self,
                input: ClassificationInput) -> ClassificationOutput:
        raw_prediction = self._raw_predict(input)
        n_pred_labels, n_pred_scores = self._postprocess_predictions(raw_prediction)
        return n_pred_labels, n_pred_scores

    def preprocess_input(self, input):
        return self.model_spec.preprocess_input(input)

    @property
    def input_size(self) -> int:
        return self.model_spec.input_size

    @property
    def class_names(self) -> List[str]:
        return self._class_names
