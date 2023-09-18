from json.decoder import JSONDecodeError
import tempfile
from pathlib import Path
from typing import List, Tuple, Callable, Union, Type, Literal

import requests
import numpy as np
import fsspec
from pathy import Pathy

from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file
from cv_pipeliner.inference_models.embedder.core import EmbedderInput, EmbedderModel, EmbedderModelSpec, EmbedderOutput
from cv_pipeliner.utils.files import copy_files_from_directory_to_temp_directory
from cv_pipeliner.utils.images import get_image_b64


class TensorFlow_EmbedderModelSpec(EmbedderModelSpec):
    input_size: Union[Tuple[int, int], List[int]]
    model_path: Union[str, Pathy]  # can be also tf.keras.Model
    saved_model_type: Literal["tf.saved_model", "tf.keras", "tf.keras.Model", "tflite", "tflite_one_image_per_batch"]
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None

    @property
    def inference_model_cls(self) -> Type["Tensorflow_EmbedderModel"]:
        from cv_pipeliner.inference_models.embedder.tensorflow import Tensorflow_EmbedderModel

        return Tensorflow_EmbedderModel


class TensorFlow_EmbedderModelSpec_TFServing(EmbedderModelSpec):
    url: str
    input_type: Literal["image_tensor", "float_image_tensor", "encoded_image_string_tensor"]
    input_name: str
    input_size: Union[Tuple[int, int], List[int]]
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None

    @property
    def inference_model_cls(self) -> Type["Tensorflow_EmbedderModel"]:
        from cv_pipeliner.inference_models.embedder.tensorflow import Tensorflow_EmbedderModel

        return Tensorflow_EmbedderModel


INPUT_TYPE_TO_DTYPE = {
    "image_tensor": np.uint8,
    "float_image_tensor": np.float32,
    "encoded_image_string_tensor": np.uint8,
}


class Tensorflow_EmbedderModel(EmbedderModel):
    def _load_tensorflow_embedder_model_spec(self, model_spec: TensorFlow_EmbedderModelSpec):
        import tensorflow as tf

        if model_spec.saved_model_type in ["tf.keras", "tf.saved_model", "tflite", "tflite_one_image_per_batch"]:
            model_openfile = fsspec.open(model_spec.model_path, "rb")
            if model_openfile.fs.isdir(model_openfile.path):
                temp_folder = copy_files_from_directory_to_temp_directory(directory=model_spec.model_path)
                model_path = Pathy.fluid(temp_folder.name)
                temp_files_cleanup = temp_folder.cleanup
            else:
                temp_file = tempfile.NamedTemporaryFile()
                with model_openfile as src:
                    temp_file.write(src.read())
                model_path = Pathy.fluid(temp_file.name)
                temp_files_cleanup = temp_file.close

            if model_spec.saved_model_type in "tf.keras":
                self.model = tf.keras.models.load_model(str(model_path), compile=False)
                self.input_dtype = np.float32
            elif model_spec.saved_model_type == "tf.saved_model":
                self.loaded_model = tf.saved_model.load(str(model_path))  # to protect from gc
                self.model = self.loaded_model.signatures["serving_default"]
                self.input_dtype = np.float32
            elif model_spec.saved_model_type in ["tflite", "tflite_one_image_per_batch"]:
                self.model = tf.lite.Interpreter(str(model_path))
                input_details = self.model.get_input_details()[0]
                self.input_index = input_details["index"]
                self.input_dtype = input_details["dtype"]
                self.output_index = self.model.get_output_details()[0]["index"]

            temp_files_cleanup()

        elif model_spec.saved_model_type == "tf.keras.Model":
            self.model = model_spec.model_path
            self.input_dtype = np.float32
        else:
            raise ValueError(
                "Tensorflow_EmbedderModel got unknown saved_model_type "
                f"in TensorFlow_EmbedderModelSpec: {model_spec.saved_model_type}"
            )

    def __init__(self, model_spec: Union[TensorFlow_EmbedderModelSpec, TensorFlow_EmbedderModelSpec_TFServing]):
        super().__init__(model_spec)

        if isinstance(model_spec, TensorFlow_EmbedderModelSpec):
            self._load_tensorflow_embedder_model_spec(model_spec)
            self._raw_predict = self._raw_predict_tensorflow
        elif isinstance(model_spec, TensorFlow_EmbedderModelSpec_TFServing):
            self.input_dtype = INPUT_TYPE_TO_DTYPE[model_spec.input_type]
            # Wake up the service
            try:
                self._raw_predict_kfserving(images=np.zeros((1, *self.input_size, 3)), timeout=1.0)
            except requests.exceptions.ReadTimeout:
                pass
            self._raw_predict = self._raw_predict_kfserving
        else:
            raise ValueError(f"Tensorflow_EmbedderModel got unknown EmbedderModelSpec: {type(model_spec)}")

        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(script_file=model_spec.preprocess_input)
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

    def _raw_predict_tensorflow(self, images: np.ndarray):
        import tensorflow as tf

        if self.model_spec.saved_model_type == "tf.saved_model":
            input_tensor = tf.convert_to_tensor(images, dtype=self.input_dtype)
            raw_predictions_batch = self.model(input_tensor)
            if isinstance(raw_predictions_batch, dict):
                key = list(raw_predictions_batch)[0]
                raw_predictions_batch = np.array(raw_predictions_batch[key])
        elif self.model_spec.saved_model_type in ["tf.keras", "tf.keras.Model"]:
            if len(images) > 0:
                raw_predictions_batch = self.model.predict(images)
            else:
                raw_predictions_batch = []
        elif self.model_spec.saved_model_type == "tflite":
            images = tf.convert_to_tensor(np.array(images), dtype=self.input_dtype)
            self.model.resize_tensor_input(0, [len(images), *self.input_size, 3])
            self.model.allocate_tensors()
            self.model.set_tensor(self.input_index, images)
            self.model.invoke()
            raw_predictions_batch = self.model.get_tensor(self.output_index)
        elif self.model_spec.saved_model_type == "tflite_one_image_per_batch":
            raw_predictions_batch = []
            for image in images:
                height, width, _ = image.shape
                image_tensor = tf.convert_to_tensor(np.array([image]), dtype=self.input_dtype)
                self.model.resize_tensor_input(0, [1, height, width, 3])
                self.model.allocate_tensors()
                self.model.set_tensor(self.input_index, image_tensor)
                self.model.invoke()
                raw_predictions_batch.append(self.model.get_tensor(self.output_index)[0])

        raw_predictions_batch = np.array(raw_predictions_batch)
        return raw_predictions_batch

    def _raw_predict_kfserving(self, images: np.ndarray, timeout: Union[float, None] = None):
        if self.model_spec.input_type in ["float_image_tensor", "image_tensor"]:
            input_data = {
                "inputs": {
                    self.model_spec.input_name: [np.array(image).astype(self.input_dtype).tolist() for image in images]
                }
            }
        elif self.model_spec.input_type == "encoded_image_string_tensor":
            input_data = {
                "instances": [
                    {self.model_spec.input_name: {"b64": get_image_b64(image, "JPEG", quality=95)}} for image in images
                ]
            }
        response = requests.post(url=self.model_spec.url, json=input_data, timeout=timeout)
        try:
            output_dict = response.json()
        except JSONDecodeError:
            raise ValueError(f"Failed to decode JSON. Response content: {response.content}")
        if not response.ok:
            raise ValueError(f"Response is not ok: {response.status_code=}; {response.content=}")
        if "outputs" in output_dict:
            raw_predictions_batch = np.array(output_dict["outputs"])
        elif "predictions" in output_dict:
            raw_predictions_batch = np.array(output_dict["predictions"])

        return raw_predictions_batch

    def predict(self, input: EmbedderInput) -> EmbedderOutput:
        input = self.preprocess_input(input)
        predictions = self._raw_predict(input)
        return predictions

    def preprocess_input(self, input: EmbedderInput):
        return self._preprocess_input(input)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size
