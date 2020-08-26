from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, ClassVar
from pathlib import Path

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from object_detection.utils import config_util
from object_detection.builders import model_builder

from two_stage_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from two_stage_pipeliner.utils.images import denormalize_bboxes, cut_bboxes_from_image


@dataclass(frozen=True)
class DetectionModelSpecTF(DetectionModelSpec):
    config_path: Union[str, Path]
    checkpoint_path: Union[str, Path]
    input_size: Tuple[int, int] = (None, None)

    @property
    def inference_model(self) -> ClassVar['DetectionModelTF']:
        from two_stage_pipeliner.inference_models.detection.tf.detector import DetectionModelTF
        return DetectionModelTF


class DetectionModelTF(DetectionModel):
    def load(self, model_spec: DetectionModelSpecTF):
        assert isinstance(model_spec, DetectionModelSpecTF)
        super().load(model_spec)
        configs = config_util.get_configs_from_pipeline_file(
            pipeline_config_path=str(model_spec.config_path)
        )
        model_config = configs['model']
        self.model = model_builder.build(
            model_config=model_config, is_training=False
        )
        ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
        ckpt.restore(str(model_spec.checkpoint_path)).expect_partial()

        # Run model through a dummy image so that variables are created
        width, height = self.input_size
        if width is None:
            width = 640
        if height is None:
            height = 640
        tf_zeros = tf.zeros([1, width, height, 3])
        self._raw_predict_single_image_tf(tf_zeros)

    def _raw_predict_single_image_tf(self, input_tensor: tf.Tensor) -> Dict:
        preprocessed_image, shapes = self.model.preprocess(input_tensor)
        raw_prediction_dict = self.model.predict(preprocessed_image, shapes)
        detector_output_dict = self.model.postprocess(
            raw_prediction_dict, shapes
        )
        return detector_output_dict

    def _postprocess_prediction(
        self,
        detection_output_dict: dict,
        score_threshold: float,
        height: int,
        width: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        raw_scores = np.array(detection_output_dict["detection_scores"][0])
        raw_bboxes = np.array(detection_output_dict["detection_boxes"][0])  # (ymin, xmin, ymax, xmax)
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
            image_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
            detector_output_dict = self._raw_predict_single_image_tf(image_tensor)
            bboxes, scores = self._postprocess_prediction(
                detector_output_dict,
                score_threshold,
                height, width
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
