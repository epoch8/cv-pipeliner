from dataclasses import dataclass
from typing import Tuple, List, Union, ClassVar
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from two_stage_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from two_stage_pipeliner.utils.images import denormalize_bboxes, cut_bboxes_from_image
from skimage.transform import resize


@dataclass(frozen=True)
class DetectionModelSpecTF_tflite(DetectionModelSpec):
    input_size: Tuple[int, int]
    model_path: Union[str, Path]

    @property
    def inference_model(self) -> ClassVar['DetectionModelTF_tflite']:
        from two_stage_pipeliner.inference_models.detection.tf.detector_tflite import DetectionModelTF_tflite
        return DetectionModelTF_tflite


class DetectionModelTF_tflite(DetectionModel):
    """
    Detector class for models trained with Object Detection API.
    Only supports models exported in .pb format.
    """
    def load(self, model_spec: DetectionModelSpecTF_tflite):
        assert isinstance(model_spec, DetectionModelSpecTF_tflite)
        super().load(model_spec)
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_spec.model_path)
        )
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        output_details = self.interpreter.get_output_details()
        self.bboxes_index = output_details[0]['index']
        self.scores_index = output_details[2]['index']

    def _raw_predict_single_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = resize(image, self.model_spec.input_size)
        image = np.array(image[None, ...], dtype=np.float32)
        self.interpreter.set_tensor(self.input_index, image)
        self.interpreter.invoke()
        raw_bboxes = np.array(self.interpreter.get_tensor(self.bboxes_index))[0]  # (ymin, xmin, ymax, xmax)
        raw_bboxes = raw_bboxes[:, [1, 0, 3, 2]]  # (xmin, ymin, xmax, ymax)
        raw_scores = np.array(self.interpreter.get_tensor(self.scores_index))[0]
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
        return self.model_spec.input_size
