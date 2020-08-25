from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable, Union, ClassVar

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from two_stage_pipeliner.inference_models.classification.core import (
    ClassificationModelSpec, ClassificationModel, ClassificationInput, ClassificationOutput
)


@dataclass(frozen=True)
class ClassificationModelSpecTF(ClassificationModelSpec):
    input_size: Tuple[int, int]
    preprocess_input: Callable[[List[np.ndarray]], np.ndarray]
    class_names: int
    model_path: Union[str, Path, tf.keras.Model]

    @property
    def inference_model(self) -> ClassVar['ClassificationModelTF']:
        from two_stage_pipeliner.inference_models.classification.tf.classifier import ClassificationModelTF
        return ClassificationModelTF


class ClassificationModelTF(ClassificationModel):
    def load(self, model_spec: ClassificationModelSpecTF):
        assert isinstance(model_spec, ClassificationModelSpecTF)
        super().load(model_spec)
        if isinstance(model_spec.model_path, str) or isinstance(model_spec.model_path, Path):
            self.model = tf.keras.models.load_model(str(model_spec.model_path))
        elif isinstance(model_spec.model_path, tf.keras.Model):
            self.model = model_spec.model_path
        assert len(model_spec.class_names) == int(self.model.output.shape[-1])
        self.id_to_class_name = {
            id: class_name for id, class_name in enumerate(model_spec.class_names)
        }
        self.batch_size = 16

    def _split_chunks(self,
                      _list: np.ndarray,
                      chunk_sizes: List[int]) -> List[np.ndarray]:
        cnt = 0
        chunks = []
        for chunk_size in chunk_sizes:
            chunks.append(_list[cnt: cnt + chunk_size])
            cnt += chunk_size
        return chunks

    def _raw_predict(self,
                     images: List[List[np.ndarray]],
                     disable_tqdm: bool) -> np.ndarray:

        predictions = []

        shapes = [len(sample_batch) for sample_batch in images]
        images = np.array(
            [item for sublist in images for item in sublist]
        )

        with tqdm(total=len(images), disable=disable_tqdm) as pbar:
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
                input: ClassificationInput,
                disable_tqdm: bool = False) -> ClassificationOutput:
        raw_prediction = self._raw_predict(input, disable_tqdm=disable_tqdm)
        n_pred_labels, n_pred_scores = self._postprocess_predictions(raw_prediction)
        return n_pred_labels, n_pred_scores

    def preprocess_input(self, input: ClassificationInput):
        return [
            self.model_spec.preprocess_input(cropped_images)
            for cropped_images in input
        ]

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size

    @property
    def class_names(self) -> List[str]:
        return self.model_spec.class_names
