import json
from json.decoder import JSONDecodeError
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Type, Literal
from pathlib import Path

import numpy as np
import fsspec
import requests
from pathy import Pathy
from joblib import Parallel, delayed


from cv_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from cv_pipeliner.utils.images import denormalize_bboxes, get_image_b64, get_image_binary_format
from cv_pipeliner.utils.files import copy_files_from_directory_to_temp_directory


@dataclass
class ObjectDetectionAPI_ModelSpec(DetectionModelSpec):
    config_path: Union[str, Path]
    checkpoint_path: Union[str, Path]
    class_names: Union[None, List[str]] = None

    @property
    def inference_model_cls(self) -> Type['ObjectDetectionAPI_DetectionModel']:
        from cv_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


@dataclass
class ObjectDetectionAPI_pb_ModelSpec(DetectionModelSpec):
    saved_model_dir: Union[str, Path]
    input_type: Literal["image_tensor", "float_image_tensor", "encoded_image_string_tensor"]
    class_names: Union[None, List[str]] = None

    @property
    def inference_model_cls(self) -> Type['ObjectDetectionAPI_DetectionModel']:
        from cv_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


@dataclass
class ObjectDetectionAPI_TFLite_ModelSpec(DetectionModelSpec):
    model_path: Union[str, Path]
    bboxes_output_index: int
    scores_output_index: int
    classes_output_index: Union[None, int]
    class_names: Union[None, List[str]] = None
    input_type: Literal["image_tensor", "float_image_tensor"] = "image_tensor"

    @property
    def inference_model_cls(self) -> Type['ObjectDetectionAPI_DetectionModel']:
        from cv_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


@dataclass
class ObjectDetectionAPI_KFServing(DetectionModelSpec):
    url: str
    input_name: str
    input_type: Literal["float_image_tensor", "encoded_b64_jpeg_image_string_tensor"]
    class_names: Union[None, List[str]] = None

    @property
    def inference_model_cls(self) -> Type['ObjectDetectionAPI_DetectionModel']:
        from cv_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


class ObjectDetectionAPI_DetectionModel(DetectionModel):
    def _load_object_detection_api(self, model_spec: ObjectDetectionAPI_ModelSpec):
        import tensorflow as tf
        from object_detection.utils import config_util
        from object_detection.builders import model_builder
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)
        model_config_path = temp_dir_path / Pathy(model_spec.config_path).name
        with open(model_config_path, 'wb') as out:
            with fsspec.open(model_spec.config_path, 'rb') as src:
                out.write(src.read())
        src_checkpoint_path = Pathy(model_spec.checkpoint_path)
        checkpoint_path = temp_dir_path / src_checkpoint_path.name
        for src_file in fsspec.open_files(f"{src_checkpoint_path}*", 'rb'):
            out_file = temp_dir_path / Pathy(src_file.path).name
            with open(out_file, 'wb') as out:
                with src_file as src:
                    out.write(src.read())
        configs = config_util.get_configs_from_pipeline_file(
            pipeline_config_path=str(model_config_path)
        )
        model_config = configs['model']
        self.model = model_builder.build(
            model_config=model_config, is_training=False
        )
        ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
        ckpt.restore(str(checkpoint_path)).expect_partial()
        self.input_dtype = np.float32

        # Run model through a dummy image so that variables are created
        zeros = np.zeros([640, 640, 3])
        self._raw_predict_single_image_default(zeros)

        temp_dir.cleanup()

    def _load_object_detection_api_pb(
        self,
        model_spec: ObjectDetectionAPI_pb_ModelSpec
    ):
        import tensorflow as tf
        temp_folder = copy_files_from_directory_to_temp_directory(
            directory=model_spec.saved_model_dir
        )
        temp_folder_path = Path(temp_folder.name)
        self.loaded_model = tf.saved_model.load(str(temp_folder_path))
        self.model = self.loaded_model.signatures["serving_default"]
        if model_spec.input_type in ["image_tensor", "encoded_image_string_tensor"]:
            self.input_dtype = np.uint8
        elif model_spec.input_type == "float_image_tensor":
            self.input_dtype = np.float32
        else:
            raise ValueError(
                "input_type of ObjectDetectionAPI_pb_ModelSpec can be image_tensor, float_image_tensor "
                "or encoded_image_string_tensor."
            )

        temp_folder.cleanup()

    def _load_object_detection_api_tflite(self, model_spec: ObjectDetectionAPI_TFLite_ModelSpec):
        import tensorflow as tf
        temp_file = tempfile.NamedTemporaryFile()
        with fsspec.open(model_spec.model_path, 'rb') as src:
            temp_file.write(src.read())
        model_path = Path(temp_file.name)

        self.model = tf.lite.Interpreter(
            model_path=str(model_path)
        )
        self.model.allocate_tensors()
        self.input_index = self.model.get_input_details()[0]['index']
        output_details = self.model.get_output_details()
        self.bboxes_index = output_details[model_spec.bboxes_output_index]['index']
        self.scores_index = output_details[model_spec.scores_output_index]['index']
        self.classes_index = output_details[model_spec.classes_output_index]['index']
        if model_spec.input_type in ["image_tensor"]:
            self.input_dtype = np.uint8
        elif model_spec.input_type == "float_image_tensor":
            self.input_dtype = np.float32
        else:
            raise ValueError(
                "input_type of ObjectDetectionAPI_pb_ModelSpec can be image_tensor or float_image_tensor"
            )

        temp_file.close()

    def __init__(
        self,
        model_spec: Union[
            ObjectDetectionAPI_ModelSpec,
            ObjectDetectionAPI_pb_ModelSpec,
            ObjectDetectionAPI_TFLite_ModelSpec,
            ObjectDetectionAPI_KFServing
        ],
    ):
        super().__init__(model_spec)

        if model_spec.class_names is not None:
            if isinstance(model_spec.class_names, str) or isinstance(model_spec.class_names, Path):
                with fsspec.open(model_spec.class_names, 'r', encoding='utf-8') as out:
                    self.class_names = np.array(json.load(out))
            else:
                self.class_names = np.array(model_spec.class_names)

            if isinstance(model_spec, ObjectDetectionAPI_ModelSpec):
                self.class_names_coef = 0  # saved_model.pb returns from 0
            else:
                self.class_names_coef = -1  # saved_model.pb returns from 1
        else:
            self.class_names = None
            self.coef = -1

        if isinstance(model_spec, ObjectDetectionAPI_ModelSpec):
            self._load_object_detection_api(model_spec)
            self._raw_predict_single_image = self._raw_predict_single_image_default
        elif isinstance(model_spec, ObjectDetectionAPI_pb_ModelSpec):
            self._load_object_detection_api_pb(model_spec)
            self._raw_predict_single_image = self._raw_predict_single_image_default
        elif isinstance(model_spec, ObjectDetectionAPI_TFLite_ModelSpec):
            self._load_object_detection_api_tflite(model_spec)
            self._raw_predict_single_image = self._raw_predict_single_image_tflite
        elif isinstance(model_spec, ObjectDetectionAPI_KFServing):
            # Wake up the service
            try:
                self._raw_predict_single_image_kfserving(
                    image=np.zeros((128, 128, 3)),
                    timeout=1.
                )
            except requests.exceptions.ReadTimeout:
                pass
            self._raw_predict_single_image = self._raw_predict_single_image_kfserving
        else:
            raise ValueError(
                f"ObjectDetectionAPI_Model got unknown DetectionModelSpec: {type(model_spec)}"
            )

    def _parse_detection_output_dict(
        self,
        detection_output_dict: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_bboxes = detection_output_dict["detection_boxes"][0]  # (ymin, xmin, ymax, xmax)
        raw_bboxes = np.array(raw_bboxes)[:, [1, 0, 3, 2]]  # (xmin, ymin, xmax, ymax)
        raw_keypoints = np.array([]).reshape(len(raw_bboxes), 0, 2)
        raw_scores = detection_output_dict["detection_scores"][0]
        raw_scores = np.array(raw_scores)
        raw_classes = detection_output_dict["detection_classes"][0]
        raw_classes = np.array(raw_classes)
        return raw_bboxes, raw_keypoints, raw_scores, raw_classes

    def _raw_predict_single_image_default(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import tensorflow as tf

        input_tensor = tf.convert_to_tensor(image, dtype=self.input_dtype)
        if (
            isinstance(self.model_spec, ObjectDetectionAPI_pb_ModelSpec)
            and
            self.model_spec.input_type == "encoded_image_string_tensor"
        ):
            input_tensor = tf.io.encode_jpeg(input_tensor, quality=100)
        input_tensor = input_tensor[None, ...]
        detection_output_dict = self.model(input_tensor)

        return self._parse_detection_output_dict(detection_output_dict)

    def _raw_predict_single_image_tflite(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        height, width, _ = image.shape
        image = np.array(image[None, ...], dtype=self.input_dtype)
        self.model.resize_tensor_input(0, [1, height, width, 3])
        self.model.allocate_tensors()
        self.model.set_tensor(self.input_index, image)
        self.model.invoke()

        raw_bboxes = np.array(self.model.get_tensor(self.bboxes_index))[0]  # (ymin, xmin, ymax, xmax)
        raw_keypoints = np.array([]).reshape(len(raw_bboxes), 0, 2)
        raw_bboxes = raw_bboxes[:, [1, 0, 3, 2]]  # (xmin, ymin, xmax, ymax)
        raw_scores = np.array(self.model.get_tensor(self.scores_index))[0]
        raw_classes = np.array(self.model.get_tensor(self.classes_index))[0]

        return raw_bboxes, raw_keypoints, raw_scores, raw_classes

    def _raw_predict_single_image_kfserving(
        self,
        image: np.ndarray,
        timeout: Union[float, None] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model_spec.input_type == "float_image_tensor":
            input_data = {
                'inputs': {
                    self.model_spec.input_name: [np.array(image).astype(np.uint8).tolist()]
                }
            }
        elif self.model_spec.input_type == "encoded_b64_jpeg_image_string_tensor":
            input_data = {
                'instances': [{
                    'input_tensor': {
                        'b64': get_image_b64(image, 'JPEG', quality=95)
                    }
                }]
            }
        response = requests.post(
            url=self.model_spec.url,
            json=input_data,
            timeout=timeout
        )
        try:
            detection_output_dict = response.json()
        except JSONDecodeError:
            raise ValueError(f"Failed to decode JSON. Response content: {response.content}")
        if not response.ok:
            raise ValueError(f"Response is not ok: {response.status_code=}; {response.content=}")
        detection_output_dict = detection_output_dict['outputs']

        return self._parse_detection_output_dict(detection_output_dict)

    def _postprocess_prediction(
        self,
        raw_bboxes: np.ndarray,
        raw_keypoints: np.ndarray,
        raw_scores: np.ndarray,
        raw_classes: np.ndarray,
        score_threshold: float,
        width: int,
        height: int,
        classification_top_n: int
    ) -> Tuple[
        List[Tuple[int, int, int, int]],
        List[List[Tuple[int, int]]],
        List[float], List[List[str]], List[List[float]]
    ]:

        raw_bboxes = denormalize_bboxes(raw_bboxes, width, height)
        mask = raw_scores > score_threshold
        bboxes = raw_bboxes[mask]
        keypoints = raw_keypoints[mask]
        scores = raw_scores[mask]
        classes = raw_classes[mask]

        correct_non_repeated_bboxes_idxs = []
        bboxes_set = set()
        for idx, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox
            if xmax - xmin > 0 and ymax - ymin > 0 and (xmin, ymin, xmax, ymax) not in bboxes_set:
                bboxes_set.add((xmin, ymin, xmax, ymax))
                correct_non_repeated_bboxes_idxs.append(idx)

        bboxes = bboxes[correct_non_repeated_bboxes_idxs]
        keypoints = keypoints[correct_non_repeated_bboxes_idxs]
        scores = scores[correct_non_repeated_bboxes_idxs]
        classes = classes[correct_non_repeated_bboxes_idxs]
        classes_scores = scores.copy()
        if self.class_names is not None:
            class_names_top_n = np.array([
                [class_name for i in range(classification_top_n)]
                for class_name in self.class_names[(classes.astype(np.int32) + self.class_names_coef)]
            ])
            classes_scores_top_n = np.array([
                [score for _ in range(classification_top_n)]
                for score in classes_scores
            ])
        else:
            class_names_top_n = np.array([
                [None for _ in range(classification_top_n)]
                for _ in classes
            ])
            classes_scores_top_n = np.array([
                [score for _ in range(classification_top_n)]
                for score in classes_scores
            ])

        return bboxes, keypoints, scores, class_names_top_n, classes_scores_top_n

    def _predict_single_image(
        self,
        image: np.ndarray,
        score_threshold: float,
        classification_top_n: int = 1
    ) -> DetectionOutput:
        height, width, _ = image.shape
        raw_bboxes, raw_keypoints, raw_scores, raw_classes = self._raw_predict_single_image(image)
        bboxes, keypoints, scores, class_names_top_k, classes_scores_top_k = self._postprocess_prediction(
            raw_bboxes=raw_bboxes,
            raw_keypoints=raw_keypoints,
            raw_scores=raw_scores,
            raw_classes=raw_classes,
            score_threshold=score_threshold,
            width=width,
            height=height,
            classification_top_n=classification_top_n
        )
        return bboxes, keypoints, scores, class_names_top_k, classes_scores_top_k

    def predict(
        self,
        input: DetectionInput,
        score_threshold: float,
        classification_top_n: int = 1,
        n_jobs: int = 1
    ) -> DetectionOutput:
        input = self.preprocess_input(input)

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._predict_single_image)(
                image=image,
                score_threshold=score_threshold,
                classification_top_n=classification_top_n,
            )
            for image in input
        )
        (
            n_pred_bboxes, n_pred_keypoints, n_pred_scores,
            n_pred_class_names_top_k, n_pred_scores_top_k
        ) = ([res[i] for res in results] for i in range(5))
        return n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k

    def preprocess_input(self, input: DetectionInput):
        return input

    @property
    def input_size(self) -> Tuple[int, int]:
        return (None, None)
