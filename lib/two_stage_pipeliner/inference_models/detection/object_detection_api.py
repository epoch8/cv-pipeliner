from dataclasses import dataclass
from typing import List, Tuple, Union, ClassVar, Literal
from pathlib import Path

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from object_detection.utils import config_util
from object_detection.builders import model_builder
from skimage.transform import resize

from two_stage_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from two_stage_pipeliner.utils.images import denormalize_bboxes, cut_bboxes_from_image


@dataclass(frozen=True)
class ObjectDetectionAPI_ModelSpec(DetectionModelSpec):
    config_path: Union[str, Path]
    checkpoint_path: Union[str, Path]

    @property
    def inference_model(self) -> ClassVar['ObjectDetectionAPI_DetectionModel']:
        from two_stage_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


@dataclass(frozen=True)
class ObjectDetectionAPI_pb_ModelSpec(DetectionModelSpec):
    saved_model_dir: Union[str, Path]
    input_type: Literal["image_tensor", "float_image_tensor", "encoded_image_string_tensor"]

    @property
    def inference_model(self) -> ClassVar['ObjectDetectionAPI_DetectionModel']:
        from two_stage_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


@dataclass(frozen=True)
class ObjectDetectionAPI_TFLite_ModelSpec(DetectionModelSpec):
    model_path: Union[str, Path]
    bboxes_output_index: int
    scores_output_index: int

    @property
    def inference_model(self) -> ClassVar['ObjectDetectionAPI_DetectionModel']:
        from two_stage_pipeliner.inference_models.detection.object_detection_api import ObjectDetectionAPI_DetectionModel
        return ObjectDetectionAPI_DetectionModel


class ObjectDetectionAPI_DetectionModel(DetectionModel):
    def _load_object_detection_api(self, model_spec: ObjectDetectionAPI_ModelSpec):
        configs = config_util.get_configs_from_pipeline_file(
            pipeline_config_path=str(model_spec.config_path)
        )
        model_config = configs['model']
        self.model = model_builder.build(
            model_config=model_config, is_training=False
        )
        ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
        ckpt.restore(str(model_spec.checkpoint_path)).expect_partial()
        self.input_dtype = tf.dtypes.float32

    def _load_object_detection_api_pb(self, model_spec: ObjectDetectionAPI_pb_ModelSpec):
        self.loaded_model = tf.saved_model.load(str(model_spec.saved_model_dir))
        self.model = self.loaded_model.signatures["serving_default"]
        if model_spec.input_type in ["image_tensor", "encoded_image_string_tensor"]:
            self.input_dtype = tf.dtypes.uint8
        elif model_spec.input_type == "float_image_tensor":
            self.input_dtype = tf.dtypes.float32
        else:
            raise ValueError(
                "input_type of ObjectDetectionAPI_pb_ModelSpec can be image_tensor, float_image_tensor "
                "or encoded_image_string_tensor."
            )

    def _load_object_detection_api_tflite(self, model_spec: ObjectDetectionAPI_TFLite_ModelSpec):
        assert isinstance(model_spec, ObjectDetectionAPI_TFLite_ModelSpec)
        super().load(model_spec)
        self.model = tf.lite.Interpreter(
            model_path=str(model_spec.model_path)
        )
        self.model.allocate_tensors()
        self.input_index = self.model.get_input_details()[0]['index']
        output_details = self.model.get_output_details()
        self.bboxes_index = output_details[model_spec.bboxes_output_index]['index']
        self.scores_index = output_details[model_spec.scores_output_index]['index']

    def load(
        self,
        model_spec: Literal[
            ObjectDetectionAPI_ModelSpec,
            ObjectDetectionAPI_pb_ModelSpec,
            ObjectDetectionAPI_TFLite_ModelSpec
        ]
    ):
        super().load(model_spec)
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

        # Run model through a dummy image so that variables are created
        zeros = np.zeros([640, 640, 3])
        self._raw_predict_single_image(zeros)

    def _raw_predict_single_image_default(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        input_tensor = tf.convert_to_tensor(image, dtype=self.input_dtype)
        if self.input_dtype == "encoded_image_string_tensor":
            input_tensor = tf.io.encode_jpeg(input_tensor)
        input_tensor = input_tensor[None, ...]
        detection_output_dict = self.model(input_tensor)

        raw_bboxes = detection_output_dict["detection_boxes"][0]  # (ymin, xmin, ymax, xmax)
        raw_bboxes = np.array(raw_bboxes)[:, [1, 0, 3, 2]]  # (xmin, ymin, xmax, ymax)
        raw_scores = detection_output_dict["detection_scores"][0]
        raw_scores = np.array(raw_scores)
        return raw_bboxes, raw_scores

    def _raw_predict_single_image_tflite(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        height, width, _ = image.shape
        image = np.array(image[None, ...], dtype=np.float32)
        self.model.resize_tensor_input(0, [1, height, width, 3])
        self.model.allocate_tensors()
        self.model.set_tensor(self.input_index, image)
        self.model.invoke()

        raw_bboxes = np.array(self.model.get_tensor(self.bboxes_index))[0]  # (ymin, xmin, ymax, xmax)
        raw_bboxes = raw_bboxes[:, [1, 0, 3, 2]]  # (xmin, ymin, xmax, ymax)
        raw_scores = np.array(self.model.get_tensor(self.scores_index))[0]

        return raw_bboxes, raw_scores

    def _postprocess_prediction(
        self,
        raw_bboxes: np.ndarray,
        raw_scores: np.ndarray,
        score_threshold: float,
        height: int,
        width: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:

        raw_bboxes = denormalize_bboxes(raw_bboxes, width, height)
        mask = raw_scores > score_threshold
        bboxes = raw_bboxes[mask]
        scores = raw_scores[mask]
        return bboxes, scores

    def predict(
        self,
        input: DetectionInput,
        score_threshold: float,
        crop_detections_from_image: bool = True,
        disable_tqdm: bool = False
    ) -> DetectionOutput:
        n_pred_cropped_images, n_pred_bboxes, n_pred_scores = [], [], []

        for image in tqdm(input, disable=disable_tqdm):
            height, width, _ = image.shape
            raw_bboxes, raw_scores = self._raw_predict_single_image(image)
            bboxes, scores = self._postprocess_prediction(
                raw_bboxes=raw_bboxes,
                raw_scores=raw_scores,
                score_threshold=score_threshold,
                height=height,
                width=width
            )

            n_pred_bboxes.append(bboxes)
            n_pred_scores.append(scores)

            if crop_detections_from_image:
                img_boxes = cut_bboxes_from_image(image, bboxes)
                n_pred_cropped_images += [img_boxes]
            else:
                n_pred_cropped_images += [[None] * len(bboxes)]

        return n_pred_cropped_images, n_pred_bboxes, n_pred_scores

    def preprocess_input(self, input: DetectionInput):
        return input

    @property
    def input_size(self) -> Tuple[int, int]:
        return (None, None)
