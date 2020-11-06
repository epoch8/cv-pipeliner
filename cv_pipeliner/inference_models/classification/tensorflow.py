import json
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable, Union, Type, Literal

import numpy as np
import tensorflow as tf

from cv_pipeliner.inference_models.classification.core import (
    ClassificationModelSpec, ClassificationModel, ClassificationInput, ClassificationOutput
)


@dataclass(frozen=True)
class TensorFlow_ClassificationModelSpec(ClassificationModelSpec):
    input_size: Union[Tuple[int, int], List[int]]
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path]
    class_names: Union[List[str], str, Path]
    model_path: Union[str, Path, tf.keras.Model]
    saved_model_type: Literal["tf.saved_model", "tf.keras", "Type[tf.keras.Model]", "tflite"]

    @property
    def inference_model_cls(self) -> Type['Tensorflow_ClassificationModel']:
        from cv_pipeliner.inference_models.classification.tensorflow import Tensorflow_ClassificationModel
        return Tensorflow_ClassificationModel


class Tensorflow_ClassificationModel(ClassificationModel):
    def _get_preprocess_input_from_script_file(
        self,
        script_file: Union[str, Path]
    ) -> Callable[[List[np.ndarray]], np.ndarray]:
        script_file = Path(script_file)
        sys.path.append(str(script_file.parent.absolute()))
        module = importlib.import_module(script_file.stem)
        sys.path.pop()
        return module.preprocess_input

    def __init__(
        self,
        model_spec: TensorFlow_ClassificationModelSpec
    ):
        assert isinstance(model_spec, TensorFlow_ClassificationModelSpec)
        super().__init__(model_spec)
        if isinstance(model_spec.class_names, str) or isinstance(model_spec.class_names, Path):
            with open(model_spec.class_names, 'r', encoding='utf-8') as out:
                self._class_names = json.load(out)
        else:
            self._class_names = model_spec.class_names
        if model_spec.saved_model_type == "tf.keras":
            self.model = tf.keras.models.load_model(str(model_spec.model_path))
        elif model_spec.saved_model_type == "tf.saved_model":
            self.loaded_model = tf.saved_model.load(str(model_spec.model_path))  # only to protect from gc
            self.model = self.loaded_model.signatures["serving_default"]
        elif model_spec.saved_model_type == "ClassType[tf.keras.Model]":
            self.model = model_spec.model_path
        elif model_spec.saved_model_type == 'tflite':
            self.model = tf.lite.Interpreter(
                model_path=str(model_spec.model_path)
            )
            self.model.allocate_tensors()
            self.input_index = self.model.get_input_details()[0]['index']
            self.output_index = self.model.get_output_details()[0]['index']
        else:
            raise ValueError(
                "Tensorflow_ClassificationModel got unknown saved_model_type "
                f"in TensorFlow_ClassificationModelSpec: {self.saved_model_type}"
            )

        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = self._get_preprocess_input_from_script_file(model_spec.preprocess_input)
        else:
            self._preprocess_input = model_spec.preprocess_input

        self.id_to_class_name = np.array([class_name for class_name in self._class_names])

        # Run model through a dummy image so that variables are created
        width, height = self.input_size
        zeros = np.zeros([1, width, height, 3], dtype=np.float32)
        self._raw_predict(zeros)

    def _raw_predict(
        self,
        images: np.ndarray
    ):
        if self.model_spec.saved_model_type == "tf.saved_model":
            input_tensor = tf.convert_to_tensor(images, dtype=tf.dtypes.float32)
            raw_predictions_batch = self.model(input_tensor)
            if isinstance(raw_predictions_batch, dict):
                key = list(raw_predictions_batch)[0]
                raw_predictions_batch = np.array(raw_predictions_batch[key])
        elif self.model_spec.saved_model_type in ["tf.keras", "ClassType[tf.keras.Model]"]:
            raw_predictions_batch = self.model.predict(images)
        elif self.model_spec.saved_model_type == 'tflite':
            self.model.resize_tensor_input(0, [len(images), *self.input_size, 3])
            self.model.allocate_tensors()
            self.model.set_tensor(self.input_index, images)
            self.model.invoke()
            raw_predictions_batch = self.model.get_tensor(self.output_index)
        return raw_predictions_batch

    def predict(
        self,
        input: ClassificationInput,
        top_n: int = 1
    ) -> ClassificationOutput:
        predictions = self._raw_predict(input)
        max_scores_top_n_idxs = (-np.array(predictions)).argsort(axis=1)[:, :top_n]
        id_to_class_names_repeated = np.repeat(
            a=self.id_to_class_name[None, ...],
            repeats=len(input),
            axis=0
        )
        pred_labels_top_n = np.take_along_axis(id_to_class_names_repeated, max_scores_top_n_idxs, axis=1)
        pred_scores_top_n = np.take_along_axis(predictions, max_scores_top_n_idxs, axis=1)

        return pred_labels_top_n, pred_scores_top_n

    def preprocess_input(self, input: ClassificationInput):
        return self._preprocess_input(input)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size

    @property
    def class_names(self) -> List[str]:
        return self._class_names
