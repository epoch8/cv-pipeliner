import json
import tempfile
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import fsspec
import numpy as np
import requests
from joblib import Parallel, delayed
from pathy import Pathy

from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file
from cv_pipeliner.inference_models.detection.core import (
    DetectionInput,
    DetectionModel,
    DetectionModelSpec,
    DetectionOutput,
)
from cv_pipeliner.utils.files import copy_files_from_directory_to_temp_directory
from cv_pipeliner.utils.images import denormalize_bboxes, get_image_b64


class ObjectDetectionAPI_ModelSpec(DetectionModelSpec):
    config_path: Union[str, Path]
    checkpoint_path: Union[str, Path]
    class_names: Optional[Union[List[str], str, Path]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    input_size: Union[Tuple[int, int], List[int]] = (None, None)
    device: Optional[str] = None

    @property
    def inference_model_cls(self) -> Type["ObjectDetectionAPI_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.object_detection_api import (
            ObjectDetectionAPI_DetectionModel,
        )

        return ObjectDetectionAPI_DetectionModel


class ObjectDetectionAPI_pb_ModelSpec(DetectionModelSpec):
    saved_model_dir: Union[str, Path]
    input_type: Literal["image_tensor", "float_image_tensor", "encoded_image_string_tensor"]
    class_names: Optional[Union[List[str], str, Path]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    input_size: Union[Tuple[int, int], List[int]] = (None, None)
    device: Optional[str] = None

    @property
    def inference_model_cls(self) -> Type["ObjectDetectionAPI_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.object_detection_api import (
            ObjectDetectionAPI_DetectionModel,
        )

        return ObjectDetectionAPI_DetectionModel


class ObjectDetectionAPI_TFLite_ModelSpec(DetectionModelSpec):
    model_path: Union[str, Path]
    bboxes_output_index: Union[int, str]
    scores_output_index: Union[int, str]
    classes_output_index: Union[int, str]
    multiclasses_scores_output_index: Optional[Union[int, str]] = None
    class_names: Optional[Union[List[str], str, Path]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    input_size: Union[Tuple[int, int], List[int]] = (None, None)
    device: Optional[str] = None

    @property
    def inference_model_cls(self) -> Type["ObjectDetectionAPI_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.object_detection_api import (
            ObjectDetectionAPI_DetectionModel,
        )

        return ObjectDetectionAPI_DetectionModel


class ObjectDetectionAPI_KFServing(DetectionModelSpec):
    url: str
    input_name: str
    input_type: Literal["image_tensor", "float_image_tensor", "encoded_image_string_tensor"]
    class_names: Optional[Union[List[str], str, Path]] = None

    @property
    def inference_model_cls(self) -> Type["ObjectDetectionAPI_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.object_detection_api import (
            ObjectDetectionAPI_DetectionModel,
        )

        return ObjectDetectionAPI_DetectionModel


INPUT_TYPE_TO_DTYPE = {
    "image_tensor": np.uint8,
    "float_image_tensor": np.float32,
    "encoded_image_string_tensor": np.uint8,
}


class ObjectDetectionAPI_DetectionModel(DetectionModel):
    def _load_object_detection_api(self, model_spec: ObjectDetectionAPI_ModelSpec):
        import tensorflow as tf
        from object_detection.builders import model_builder
        from object_detection.utils import config_util

        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)
        model_config_path = temp_dir_path / Pathy.fluid(model_spec.config_path).name
        with open(model_config_path, "wb") as out:
            with fsspec.open(model_spec.config_path, "rb") as src:
                out.write(src.read())
        src_checkpoint_path = Pathy.fluid(model_spec.checkpoint_path)
        checkpoint_path = temp_dir_path / src_checkpoint_path.name
        for src_file in fsspec.open_files(f"{src_checkpoint_path}*", "rb"):
            out_file = temp_dir_path / Pathy.fluid(src_file.path).name
            with open(out_file, "wb") as out:
                with src_file as src:
                    out.write(src.read())
        if model_spec.device is not None:
            self.tf_device = tf.device(model_spec.device)
            self.tf_device.__enter__()
        else:
            self.tf_device = None
        try:
            configs = config_util.get_configs_from_pipeline_file(pipeline_config_path=str(model_config_path))
            model_config = configs["model"]
            self.model = model_builder.build(model_config=model_config, is_training=False)
            ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
            ckpt.restore(str(checkpoint_path)).expect_partial()
        finally:
            if self.tf_device is not None:
                self.tf_device.__exit__()
        self.input_dtype = np.float32

        # Run model through a dummy image so that variables are created
        zeros = np.zeros([640, 640, 3])
        self._raw_predict_single_image_default(zeros)

        temp_dir.cleanup()

    def _load_object_detection_api_pb(self, model_spec: ObjectDetectionAPI_pb_ModelSpec):
        import tensorflow as tf

        temp_folder = copy_files_from_directory_to_temp_directory(directory=model_spec.saved_model_dir)
        temp_folder_path = Path(temp_folder.name)
        if model_spec.device is not None:
            self.tf_device = tf.device(model_spec.device)
            self.tf_device.__enter__()
        else:
            self.tf_device = None
        try:
            self.loaded_model = tf.saved_model.load(str(temp_folder_path))
            self.model = self.loaded_model.signatures["serving_default"]
        finally:
            if self.tf_device is not None:
                self.tf_device.__exit__()
        self.input_dtype = INPUT_TYPE_TO_DTYPE[model_spec.input_type]
        temp_folder.cleanup()

    def _load_object_detection_api_tflite(self, model_spec: ObjectDetectionAPI_TFLite_ModelSpec):
        import tensorflow as tf

        temp_file = tempfile.NamedTemporaryFile()
        with fsspec.open(model_spec.model_path, "rb") as src:
            temp_file.write(src.read())
        model_path = Path(temp_file.name)

        self.model = tf.lite.Interpreter(model_path=str(model_path))
        self.model.allocate_tensors()
        input_detail = self.model.get_input_details()[0]
        self.input_index = input_detail["index"]
        output_details = self.model.get_output_details()
        output_name_to_index = {output_detail["name"]: output_detail["index"] for output_detail in output_details}
        if isinstance(model_spec.bboxes_output_index, str):
            self.bboxes_index = output_name_to_index[model_spec.bboxes_output_index]
        else:
            self.bboxes_index = output_details[model_spec.bboxes_output_index]["index"]
        if isinstance(model_spec.bboxes_output_index, str):
            self.scores_index = output_name_to_index[model_spec.scores_output_index]
        else:
            self.scores_index = output_details[model_spec.scores_output_index]["index"]
        if isinstance(model_spec.classes_output_index, str):
            self.classes_index = output_name_to_index[model_spec.classes_output_index]
        else:
            self.classes_index = output_details[model_spec.classes_output_index]["index"]
        self.input_dtype = input_detail["dtype"]
        temp_file.close()

    def __init__(
        self,
        model_spec: Union[
            ObjectDetectionAPI_ModelSpec,
            ObjectDetectionAPI_pb_ModelSpec,
            ObjectDetectionAPI_TFLite_ModelSpec,
            ObjectDetectionAPI_KFServing,
        ],
    ):
        super().__init__(model_spec)

        if model_spec.class_names is not None:
            if isinstance(model_spec.class_names, str) or isinstance(model_spec.class_names, Path):
                with fsspec.open(model_spec.class_names, "r", encoding="utf-8") as out:
                    self.class_names = np.array(json.load(out))
            else:
                self.class_names = np.array(model_spec.class_names)

            if isinstance(model_spec, ObjectDetectionAPI_pb_ModelSpec):
                self.class_names_coef = -1  # saved_model.pb returns from 1
            else:
                self.class_names_coef = 0
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
            self.input_dtype = INPUT_TYPE_TO_DTYPE[model_spec.input_type]
            # Wake up the service
            try:
                self._raw_predict_single_image_kfserving(image=np.zeros((128, 128, 3)), timeout=1.0)
            except requests.exceptions.ReadTimeout:
                pass
            self._raw_predict_single_image = self._raw_predict_single_image_kfserving
        else:
            raise ValueError(f"ObjectDetectionAPI_Model got unknown DetectionModelSpec: {type(model_spec)}")

        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(script_file=model_spec.preprocess_input)
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

    def _parse_detection_output_dict(
        self, detection_output_dict: Dict[str, Any]
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
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import tensorflow as tf

        if self.tf_device is not None:
            self.tf_device.__enter__()
        try:
            input_tensor = tf.convert_to_tensor(image, dtype=self.input_dtype)
            if (
                isinstance(self.model_spec, ObjectDetectionAPI_pb_ModelSpec)
                and self.model_spec.input_type == "encoded_image_string_tensor"
            ):
                input_tensor = tf.io.encode_jpeg(input_tensor, quality=100)
            input_tensor = input_tensor[None, ...]
            detection_output_dict = self.model(input_tensor)
        finally:
            if self.tf_device is not None:
                self.tf_device.__exit__()

        return self._parse_detection_output_dict(detection_output_dict)

    def _raw_predict_single_image_tflite(
        self, image: np.ndarray
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
        self, image: np.ndarray, timeout: Union[float, None] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.model_spec.input_type in ["float_image_tensor", "image_tensor"]:
            input_data = {"inputs": {self.model_spec.input_name: [np.array(image).astype(self.input_dtype).tolist()]}}
        elif self.model_spec.input_type == "encoded_image_string_tensor":
            input_data = {
                "instances": [{self.model_spec.input_name: {"b64": get_image_b64(image, "JPEG", quality=95)}}]
            }
        response = requests.post(url=self.model_spec.url, json=input_data, timeout=timeout)
        try:
            detection_output_dict = response.json()
        except JSONDecodeError:
            raise ValueError(f"Failed to decode JSON. Response content: {response.content}")
        if not response.ok:
            raise ValueError(f"Response is not ok: {response.status_code=}; {response.content=}")
        if "outputs" in detection_output_dict:
            detection_output_dict = detection_output_dict["outputs"]
        elif "predictions" in detection_output_dict:
            detection_output_dict = detection_output_dict["predictions"]
            detection_output_dict = {
                "detection_boxes": [detection_output_dict[0]["detection_boxes"]],
                "detection_scores": [detection_output_dict[0]["detection_scores"]],
                "detection_classes": [detection_output_dict[0]["detection_classes"]],
            }
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
        classification_top_n: int,
    ) -> Tuple[
        List[Tuple[int, int, int, int]], List[List[Tuple[int, int]]], List[float], List[List[str]], List[List[float]]
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
            class_names_top_n = np.array(
                [
                    [class_name for i in range(classification_top_n)]
                    for class_name in self.class_names[(classes.astype(np.int32) + self.class_names_coef)]
                ]
            )
            classes_scores_top_n = np.array([[score for _ in range(classification_top_n)] for score in classes_scores])
        else:
            class_names_top_n = np.array([[None for _ in range(classification_top_n)] for _ in classes])
            classes_scores_top_n = np.array([[score for _ in range(classification_top_n)] for score in classes_scores])

        return bboxes, keypoints, scores, class_names_top_n, classes_scores_top_n

    def _predict_single_image(
        self, image: np.ndarray, score_threshold: float, classification_top_n: int = 1
    ) -> DetectionOutput:
        height, width, _ = image.shape
        input_image = self.preprocess_input([image])[0]
        raw_bboxes, raw_keypoints, raw_scores, raw_classes = self._raw_predict_single_image(input_image)
        bboxes, keypoints, scores, class_names_top_k, classes_scores_top_k = self._postprocess_prediction(
            raw_bboxes=raw_bboxes,
            raw_keypoints=raw_keypoints,
            raw_scores=raw_scores,
            raw_classes=raw_classes,
            score_threshold=score_threshold,
            width=width,
            height=height,
            classification_top_n=classification_top_n,
        )
        return bboxes, keypoints, scores, class_names_top_k, classes_scores_top_k

    def predict(
        self, input: DetectionInput, score_threshold: float, classification_top_n: int = 1, n_jobs: int = 1
    ) -> DetectionOutput:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._predict_single_image)(
                image=image,
                score_threshold=score_threshold,
                classification_top_n=classification_top_n,
            )
            for image in input
        )
        (n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k) = (
            [res[i] for res in results] for i in range(5)
        )
        n_pred_masks = [[[] for _ in pred_bboxes] for pred_bboxes in n_pred_bboxes]
        return (
            n_pred_bboxes,
            n_pred_keypoints,
            n_pred_masks,
            n_pred_scores,
            n_pred_class_names_top_k,
            n_pred_scores_top_k,
        )

    def preprocess_input(self, input: DetectionInput):
        return self._preprocess_input(input)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size
