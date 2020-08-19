from typing import List, Dict, Tuple

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from object_detection.utils import config_util
from object_detection.builders import model_builder

from two_stage_pipeliner.inference_models.detection.core import DetectionModel, DetectionInput, DetectionOutput
from two_stage_pipeliner.inference_models.detection.tf.specs import DetectorModelSpecTF
from two_stage_pipeliner.utils.images import denormalize_bboxes, cut_bboxes_from_image


class DetectorTF(DetectionModel):
    def load(self, checkpoint: DetectorModelSpecTF):
        super().load(checkpoint)
        model_spec = checkpoint
        configs = config_util.get_configs_from_pipeline_file(
            pipeline_config_path=str(model_spec.config_path)
        )
        model_config = configs['model']
        self.model = model_builder.build(
            model_config=model_config, is_training=False
        )
        ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
        ckpt_path = (
            model_spec.model_dir / 'checkpoint' / model_spec.checkpoint_filename
        )
        ckpt.restore(str(ckpt_path)).expect_partial()

        # Run model through a dummy image so that variables are created
        tf_zeros = tf.zeros(
            [1, model_spec.input_size, model_spec.input_size, 3]
        )
        self._raw_predict_single_image_tf(tf_zeros)
        self.model_spec = model_spec
        self.disable_tqdm = False

    def _raw_predict_single_image_tf(self, input_tensor: tf.Tensor) -> Dict:
        preprocessed_image, shapes = self.model.preprocess(input_tensor)
        raw_prediction_dict = self.model.predict(preprocessed_image, shapes)
        detector_output_dict = self.model.postprocess(
            raw_prediction_dict, shapes
        )
        return detector_output_dict

    def _raw_predict(self,
                     images: np.ndarray) -> List[Dict]:
        detector_output_dicts = []
        for image in tqdm(images, disable=self.disable_tqdm):
            image_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
            detector_output_dict = self._raw_predict_single_image_tf(image_tensor)
            detector_output_dicts.append(detector_output_dict)

        return detector_output_dicts

    def _postprocess_prediction(
        self,
        detection_output_dict: dict,
        score_threshold: float,
        height: int,
        width: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        raw_scores = np.array(detection_output_dict["detection_scores"][0])
        raw_bboxes = np.array(detection_output_dict["detection_boxes"][0])
        raw_bboxes = denormalize_bboxes(raw_bboxes, width, height)

        mask = raw_scores > score_threshold
        bboxes = raw_bboxes[mask]
        scores = raw_scores[mask]
        return bboxes, scores

    def predict(
        self,
        images: DetectionInput,
        score_threshold: float,
        crop_detections_from_image: bool = True,
    ) -> DetectionOutput:
        n_pred_cropped_images, n_pred_bboxes, n_pred_scores = [], [], []

        for image in tqdm(images, disable=self.disable_tqdm):
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

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size
