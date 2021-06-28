import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable, Union, Type, Literal

import requests
import numpy as np
import fsspec
from pathy import Pathy

from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file
from cv_pipeliner.inference_models.classification.core import (
    ClassificationModelSpec, ClassificationModel, ClassificationInput, ClassificationOutput
)
from cv_pipeliner.utils.files import copy_files_from_directory_to_temp_directory
from cv_pipeliner.utils.images import get_image_b64


@dataclass
class TensorFlow_ClassificationModelSpec(ClassificationModelSpec):
    input_size: Union[Tuple[int, int], List[int]]
    class_names: Union[List[str], str, Path]
    model_path: Union[str, Pathy]  # can be also tf.keras.Model
    saved_model_type: Literal["tf.saved_model", "tf.keras", "tf.keras.Model", "tflite"]
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None

    @property
    def inference_model_cls(self) -> Type['Tensorflow_ClassificationModel']:
        from cv_pipeliner.inference_models.classification.tensorflow import Tensorflow_ClassificationModel
        return Tensorflow_ClassificationModel


@dataclass
class TensorFlow_ClassificationModelSpec_TFServing(ClassificationModelSpec):
    url: str
    input_name: str
    input_size: Union[Tuple[int, int], List[int]]
    class_names: Union[List[str], str, Path]
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None

    @property
    def inference_model_cls(self) -> Type['Tensorflow_ClassificationModel']:
        from cv_pipeliner.inference_models.classification.tensorflow import Tensorflow_ClassificationModel
        return Tensorflow_ClassificationModel


class Tensorflow_ClassificationModel(ClassificationModel):
    def _load_tensorflow_classification_model_spec(
        self,
        model_spec: TensorFlow_ClassificationModelSpec
    ):
        import tensorflow as tf
        if model_spec.saved_model_type in ["tf.keras", "tf.saved_model", "tflite"]:
            model_openfile = fsspec.open(model_spec.model_path, 'rb')
            if model_openfile.fs.isdir(model_openfile.path):
                temp_folder = copy_files_from_directory_to_temp_directory(
                    directory=model_spec.model_path
                )
                model_path = Pathy(temp_folder.name)
                temp_files_cleanup = temp_folder.cleanup
            else:
                temp_file = tempfile.NamedTemporaryFile()
                with model_openfile as src:
                    temp_file.write(src.read())
                model_path = Pathy(temp_file.name)
                temp_files_cleanup = temp_file.close

            if model_spec.saved_model_type in "tf.keras":
                self.model = tf.keras.models.load_model(str(model_path), compile=False)
                self.input_dtype = np.float32
            elif model_spec.saved_model_type == "tf.saved_model":
                self.loaded_model = tf.saved_model.load(str(model_path))  # to protect from gc
                self.model = self.loaded_model.signatures["serving_default"]
                self.input_dtype = np.float32
            elif model_spec.saved_model_type == 'tflite':
                self.model = tf.lite.Interpreter(str(model_path))
                self.model.allocate_tensors()
                input_details = self.model.get_input_details()[0]
                self.input_index = input_details['index']
                self.input_dtype = input_details['dtype']
                self.output_index = self.model.get_output_details()[0]['index']

            temp_files_cleanup()

        elif model_spec.saved_model_type == "tf.keras.Model":
            self.model = model_spec.model_path
            self.input_dtype = np.float32
        else:
            raise ValueError(
                "Tensorflow_ClassificationModel got unknown saved_model_type "
                f"in TensorFlow_ClassificationModelSpec: {self.saved_model_type}"
            )

    def __init__(
        self,
        model_spec: Union[
            TensorFlow_ClassificationModelSpec,
            TensorFlow_ClassificationModelSpec_TFServing
        ]
    ):
        super().__init__(model_spec)

        if isinstance(model_spec.class_names, str) or isinstance(model_spec.class_names, Path):
            with fsspec.open(model_spec.class_names, 'r', encoding='utf-8') as out:
                self._class_names = json.load(out)
        else:
            self._class_names = model_spec.class_names

        if isinstance(model_spec, TensorFlow_ClassificationModelSpec):
            self._load_tensorflow_classification_model_spec(model_spec)
            self._raw_predict = self._raw_predict_tensorflow
        elif isinstance(model_spec, TensorFlow_ClassificationModelSpec_TFServing):
            # Wake up the service
            try:
                self._raw_predict_kfserving(
                    images=np.zeros((1, *self.input_size, 3)),
                    timeout=1.
                )
            except requests.exceptions.ReadTimeout:
                pass
            self._raw_predict = self._raw_predict_kfserving
        else:
            raise ValueError(
                f"Tensorflow_ClassificationModel got unknown ClassificationModelSpec: {type(model_spec)}"
            )

        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(
                script_file=model_spec.preprocess_input
            )
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

        self.id_to_class_name = np.array([class_name for class_name in self._class_names])

    def _raw_predict_tensorflow(
        self,
        images: np.ndarray
    ):
        import tensorflow as tf
        if self.model_spec.saved_model_type == "tf.saved_model":
            input_tensor = tf.convert_to_tensor(images, dtype=self.input_dtype)
            raw_predictions_batch = self.model(input_tensor)
            if isinstance(raw_predictions_batch, dict):
                key = list(raw_predictions_batch)[0]
                raw_predictions_batch = np.array(raw_predictions_batch[key])
        elif self.model_spec.saved_model_type in ["tf.keras", "tf.keras.Model"]:
            raw_predictions_batch = self.model.predict(images)
        elif self.model_spec.saved_model_type == 'tflite':
            images = tf.convert_to_tensor(images, dtype=self.input_dtype)
            self.model.resize_tensor_input(0, [len(images), *self.input_size, 3])
            self.model.allocate_tensors()
            self.model.set_tensor(self.input_index, images)
            self.model.invoke()
            raw_predictions_batch = self.model.get_tensor(self.output_index)
        return raw_predictions_batch

    def _raw_predict_kfserving(
        self,
        images: np.ndarray,
        timeout: Union[float, None] = None
    ):
        images_encoded_b64 = [get_image_b64(image, 'JPEG') for image in images]
        response = requests.post(
            url=self.model_spec.url,
            data=json.dumps({
                'instances': [
                    {
                        self.model_spec.input_name: {
                            'b64': image
                        }
                    }
                    for image in images_encoded_b64
                ]
            }),
            timeout=timeout
        )
        output_dict = json.loads(response.content)
        if not response.ok:
            if 'error' in output_dict:
                error = output_dict['error']
            else:
                error = response.response
            raise ValueError(error)
        raw_predictions_batch = np.array(output_dict['predictions'])

        return raw_predictions_batch

    def predict(
        self,
        input: ClassificationInput,
        top_n: int = 1
    ) -> ClassificationOutput:
        input = self.preprocess_input(input)
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
