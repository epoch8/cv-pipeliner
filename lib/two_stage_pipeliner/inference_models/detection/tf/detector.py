from typing import List, Dict, Tuple

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from object_detection.utils import config_util
from object_detection.builders import model_builder

from two_stage_pipeliner.inference_models.detection.core import DetectionModel, DetectionInput, DetectionOutput
from two_stage_pipeliner.inference_models.detection.tf.specs import DetectorModelSpecTF


class DetectorTF(DetectionModel):
    def __init__(self,
                 model_spec: DetectorModelSpecTF,
                 disable_tqdm: bool = False):
        super(DetectorTF, self).__init__()
        self.load(model_spec)
        self.disable_tqdm = disable_tqdm

    def load(self, checkpoint: DetectorModelSpecTF):
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
            image_tf = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
            detector_output_dict = self._raw_predict_single_image_tf(image_tf)
            detector_output_dicts.append(detector_output_dict)

        return detector_output_dicts

    def _denormalize_bboxes(self,
                            bboxes: List[Tuple[float, float, float, float]],
                            image_width: int,
                            image_height: int) -> List[Tuple[int, int, int, int]]:
        bboxes = np.array(bboxes.copy())
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * image_height
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * image_width
        bboxes = bboxes.round().astype(int)
        return bboxes

    def _cut_bboxes_from_image(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]]
    ) -> List[np.ndarray]:

        img_boxes = []
        for bbox in bboxes:
            ymin, xmin, ymax, xmax = bbox
            img_boxes.append(image[ymin:ymax, xmin:xmax])
        return img_boxes

    def _postprocess_predictions(
            self,
            images: DetectionInput,
            detector_output_dicts: List[Dict],
            score_threshold: float,
            open_img_boxes: bool = True
    ) -> DetectionOutput:

        assert len(images) == len(detector_output_dicts)

        n_img_boxes, n_pred_bboxes, n_pred_scores = [], [], []

        for image, detector_output_dict in tqdm(
            list(zip(images, detector_output_dicts)),
            disable=self.disable_tqdm
        ):
            width, height = image.shape[1], image.shape[0]
            raw_scores = np.array(
                detector_output_dict['detection_scores']
            )[0]
            raw_bboxes = np.array(
                detector_output_dict['detection_boxes']
            )[0]
            raw_bboxes = self._denormalize_bboxes(
                raw_bboxes, width, height
            )
            mask = raw_scores > score_threshold
            bboxes = raw_bboxes[mask]
            scores = raw_scores[mask]

            n_pred_bboxes.append(bboxes)
            n_pred_scores.append(scores)

            if open_img_boxes:
                img_boxes = self._cut_bboxes_from_image(image, bboxes)
                n_img_boxes += [img_boxes]
            else:
                n_img_boxes += [[None] * len(bboxes)]

        return n_img_boxes, n_pred_bboxes, n_pred_scores

    def predict(self,
                input: DetectionInput,
                score_threshold: float,
                open_img_boxes: bool = True) -> DetectionOutput:

        detector_output_dicts = self._raw_predict(input)
        n_img_boxes, n_pred_bboxes, n_pred_scores = self._postprocess_predictions(
            images=input,
            detector_output_dicts=detector_output_dicts,
            score_threshold=score_threshold,
            open_img_boxes=open_img_boxes,
        )
        return n_img_boxes, n_pred_bboxes, n_pred_scores

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self) -> int:
        return self.model_spec.input_size
