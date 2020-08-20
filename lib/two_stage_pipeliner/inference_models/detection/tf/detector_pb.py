from typing import Dict, Tuple, List

import tensorflow as tf
from tqdm import tqdm

from two_stage_pipeliner.inference_models.detection.core import DetectionModel, DetectionInput, DetectionOutput
from two_stage_pipeliner.inference_models.detection.tf.specs_pb import DetectorModelSpecTF_pb
from two_stage_pipeliner.utils.images import denormalize_bboxes, cut_bboxes_from_image


class DetectorTF_pb(DetectionModel):
    """
    Detector class for models trained with Object Detection API.
    Only supports models exported in .pb format.
    """
    def load(self, checkpoint: DetectorModelSpecTF_pb):
        super().load(self, checkpoint)
        model_spec = checkpoint
        self.model = tf.keras.models.load_model(str(model_spec.checkpoint_path))
        self.model = self.model.signatures["serving_default"]
        if model_spec.input_type == "image_tensor":
            self.input_dtype = tf.dtypes.uint8
        elif model_spec.input_type == "float_image_tensor":
            self.input_dtype = tf.dtypes.float32
        else:
            raise ValueError("input_type of DetectorModelSpecTF_pb can be image_tensor or float_image_tensor.")
        # Run model through a dummy image so that variables are created
        tf_zeros = tf.zeros(
            [1, model_spec.input_size, model_spec.input_size, 3],
            dtype=self.input_dtype
        )

        self._raw_predict_single_image_tf(tf_zeros)
        self.model_spec = model_spec
        self.disable_tqdm = False

    def _raw_predict_single_image_tf(self, input_tensor: tf.Tensor) -> Dict:
        output_dict = self.model(input_tensor)
        num_detections = int(output_dict.pop("num_detections"))
        output_dict = {
            key: value[0, :num_detections].numpy() for key, value in output_dict.items()
        }
        return output_dict

    def _postprocess_prediction(
        self,
        detection_output_dict: dict,
        score_threshold: float,
        height: int,
        width: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int]]:
        raw_scores = detection_output_dict["detection_scores"]
        raw_bboxes = detection_output_dict["detection_boxes"]
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
            image_tensor = tf.convert_to_tensor(image[None, ...], dtype=self.input_dtype)
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
