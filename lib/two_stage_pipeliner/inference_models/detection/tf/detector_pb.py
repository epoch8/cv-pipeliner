from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Literal, ClassVar
from pathlib import Path

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from two_stage_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from two_stage_pipeliner.utils.images import denormalize_bboxes, cut_bboxes_from_image


@dataclass(frozen=True)
class DetectionModelSpecTF_pb(DetectionModelSpec):
    saved_model_dir: Union[str, Path]
    input_type: Literal["image_tensor", "float_image_tensor", "encoded_image_string_tensor"]
    input_size: Tuple[int, int] = (None, None)

    @property
    def inference_model(self) -> ClassVar['DetectionModelTF_pb']:
        from two_stage_pipeliner.inference_models.detection.tf.detector_pb import DetectionModelTF_pb
        return DetectionModelTF_pb


class DetectionModelTF_pb(DetectionModel):
    """
    Detector class for models trained with Object Detection API.
    Only supports models exported in .pb format.
    """
    def load(self, model_spec: DetectionModelSpecTF_pb):
        assert isinstance(model_spec, DetectionModelSpecTF_pb)
        super().load(model_spec)
        self.loaded_model = tf.saved_model.load(str(model_spec.saved_model_dir))
        self.model = self.loaded_model.signatures["serving_default"]
        if model_spec.input_type in ["image_tensor", "encoded_image_string_tensor"]:
            self.input_dtype = tf.dtypes.uint8
        elif model_spec.input_type == "float_image_tensor":
            self.input_dtype = tf.dtypes.float32
        else:
            raise ValueError(
                "input_type of DetectionModelSpecTF_pb can be image_tensor, float_image_tensor "
                "or encoded_image_string_tensor."
            )

        # Run model through a dummy image so that variables are created
        width, height = model_spec.input_size
        if width is None:
            width = 640
        if height is None:
            height = 640
        zeros = np.zeros([width, height, 3])
        self._raw_predict_single_image(zeros)

    def _raw_predict_single_image(self, image: np.ndarray) -> Dict:
        input_tensor = tf.convert_to_tensor(image, dtype=self.input_dtype)
        if self.model_spec.input_type == "encoded_image_string_tensor":
            input_tensor = tf.io.encode_jpeg(input_tensor)
        input_tensor = input_tensor[None, ...]
        output_dict = self.model(input_tensor)
        num_detections = int(output_dict.pop("num_detections"))
        detection_output_dict = {
            key: value[0, :num_detections].numpy() for key, value in output_dict.items()
        }
        return detection_output_dict

    def _postprocess_prediction(
        self,
        detection_output_dict: dict,
        score_threshold: float,
        height: int,
        width: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        raw_scores = detection_output_dict["detection_scores"]
        raw_bboxes = detection_output_dict["detection_boxes"]  # (ymin, xmin, ymax, xmax)
        raw_bboxes = raw_bboxes[:, [1, 0, 3, 2]]  # (xmin, ymin, xmax, ymax)
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
            detection_output_dict = self._raw_predict_single_image(image)
            bboxes, scores = self._postprocess_prediction(
                detection_output_dict=detection_output_dict,
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
        return self.model_spec.input_size
