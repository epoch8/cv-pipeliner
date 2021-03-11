import json
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Union, Type, Literal
from pathlib import Path

import tensorflow as tf
import numpy as np
import fsspec
from pathy import Pathy

from cv_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from cv_pipeliner.utils.images import denormalize_bboxes
from cv_pipeliner.utils.files import copy_files_from_directory_to_temp_directory


@dataclass
class ObjectDetectionAPI_ModelSpec(DetectionModelSpec):
    config_path: Union[str, Path]
    checkpoint_path: Union[str, Path]
    class_names: List[str] = None

    @property
    def inference_model_cls(self) -> Type['ObjectDetectionAPI_DetectionModel']:
        from cv_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


@dataclass
class ObjectDetectionAPI_pb_ModelSpec(DetectionModelSpec):
    saved_model_dir: Union[str, Path]
    input_type: Literal["image_tensor", "float_image_tensor", "encoded_image_string_tensor"]
    class_names: List[str] = None

    @property
    def inference_model_cls(self) -> Type['ObjectDetectionAPI_DetectionModel']:
        from cv_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


@dataclass
class ObjectDetectionAPI_TFLite_ModelSpec(DetectionModelSpec):
    model_path: Union[str, Path]
    bboxes_output_index: int
    scores_output_index: int
    classes_output_index: int = None
    class_names: List[str] = None

    @property
    def inference_model_cls(self) -> Type['ObjectDetectionAPI_DetectionModel']:
        from cv_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


class ObjectDetectionAPI_DetectionModel(DetectionModel):
    def _load_object_detection_api(self, model_spec: ObjectDetectionAPI_ModelSpec):
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

        temp_file.close()

    def __init__(
        self,
        model_spec: Union[
            ObjectDetectionAPI_ModelSpec,
            ObjectDetectionAPI_pb_ModelSpec,
            ObjectDetectionAPI_TFLite_ModelSpec
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
        else:
            raise ValueError(
                f"ObjectDetectionAPI_Model got unknown DetectionModelSpec: {type(model_spec)}"
            )

    def _raw_predict_single_image_default(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = tf.convert_to_tensor(image, dtype=self.input_dtype)
        if (
            isinstance(self.model_spec, ObjectDetectionAPI_pb_ModelSpec)
            and
            self.model_spec.input_type == "encoded_image_string_tensor"
        ):
            input_tensor = tf.io.encode_jpeg(input_tensor)
        input_tensor = input_tensor[None, ...]
        detection_output_dict = self.model(input_tensor)

        raw_bboxes = detection_output_dict["detection_boxes"][0]  # (ymin, xmin, ymax, xmax)
        raw_bboxes = np.array(raw_bboxes)[:, [1, 0, 3, 2]]  # (xmin, ymin, xmax, ymax)
        raw_scores = detection_output_dict["detection_scores"][0]
        raw_scores = np.array(raw_scores)
        raw_classes = detection_output_dict["detection_classes"][0]
        raw_classes = np.array(raw_classes)

        return raw_bboxes, raw_scores, raw_classes

    def _raw_predict_single_image_tflite(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width, _ = image.shape
        image = np.array(image[None, ...], dtype=np.float32)
        self.model.resize_tensor_input(0, [1, height, width, 3])
        self.model.allocate_tensors()
        self.model.set_tensor(self.input_index, image)
        self.model.invoke()

        raw_bboxes = np.array(self.model.get_tensor(self.bboxes_index))[0]  # (ymin, xmin, ymax, xmax)
        raw_bboxes = raw_bboxes[:, [1, 0, 3, 2]]  # (xmin, ymin, xmax, ymax)
        raw_scores = np.array(self.model.get_tensor(self.scores_index))[0]
        raw_classes = np.array(self.model.get_tensor(self.scores_index))[0]

        return raw_bboxes, raw_scores, raw_classes

    def _postprocess_prediction(
        self,
        raw_bboxes: np.ndarray,
        raw_scores: np.ndarray,
        raw_classes: np.ndarray,
        score_threshold: float,
        height: int,
        width: int,
        classification_top_n: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[List[str]], List[List[float]]]:

        raw_bboxes = denormalize_bboxes(raw_bboxes, width, height)
        mask = raw_scores > score_threshold
        bboxes = raw_bboxes[mask]
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

        return bboxes, scores, class_names_top_n, classes_scores_top_n

    def predict(
        self,
        input: DetectionInput,
        score_threshold: float,
        classification_top_n: int = 1
    ) -> DetectionOutput:
        n_pred_bboxes, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k = [], [], [], []

        for image in input:
            height, width, _ = image.shape
            raw_bboxes, raw_scores, raw_classes = self._raw_predict_single_image(image)
            bboxes, scores, class_names_top_k, classes_scores_top_k = self._postprocess_prediction(
                raw_bboxes=raw_bboxes,
                raw_scores=raw_scores,
                raw_classes=raw_classes,
                score_threshold=score_threshold,
                height=height,
                width=width,
                classification_top_n=classification_top_n
            )

            n_pred_bboxes.append(bboxes)
            n_pred_scores.append(scores)
            n_pred_class_names_top_k.append(class_names_top_k)
            n_pred_scores_top_k.append(classes_scores_top_k)

        return n_pred_bboxes, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k

    def preprocess_input(self, input: DetectionInput):
        return input

    @property
    def input_size(self) -> Tuple[int, int]:
        return (None, None)
