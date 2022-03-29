import json
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Callable
from pathlib import Path

import cv2
import numpy as np
import fsspec

from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file
from cv_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from cv_pipeliner.utils.images import denormalize_bboxes


@dataclass
class YOLOv5_ModelSpec(DetectionModelSpec):
    """
    note: model_path can be set as torch.hub.load('ultralytics/yolov5', 'yolov5s')
    """
    model_path: Optional[Union[str, Path, 'torch.nn.Module']]  # noqa: F821
    class_names: Optional[List[str]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    input_size: Union[Tuple[int, int], List[int]] = (None, None)

    @property
    def inference_model_cls(self) -> Type['YOLOv5_DetectionModel']:
        from cv_pipeliner.inference_models.detection.yolov5 import YOLOv5_DetectionModel
        return YOLOv5_DetectionModel


@dataclass
class YOLOv5_TFLite_ModelSpec(DetectionModelSpec):
    """
    note: model_path can be set as torch.hub.load('ultralytics/yolov5', 'yolov5s')
    """
    model_path: Union[str, Path]
    input_size: Union[Tuple[int, int], List[int]]
    class_names: Optional[List[str]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None


    @property
    def inference_model_cls(self) -> Type['YOLOv5_DetectionModel']:
        from cv_pipeliner.inference_models.detection.yolov5 import YOLOv5_DetectionModel
        return YOLOv5_DetectionModel


class YOLOv5_DetectionModel(DetectionModel):
    def _load_yolov5_model(self, model_spec: YOLOv5_ModelSpec):
        import torch

        if isinstance(model_spec.model_path, torch.nn.Module):
            self.model = model_spec.model_path
            return

        temp_file = tempfile.NamedTemporaryFile(suffix='.pt')
        with fsspec.open(model_spec.model_path, 'rb') as src:
            temp_file.write(src.read())
        model_path_tmp = Path(temp_file.name)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path_tmp))
        temp_file.close()

    def __init__(
        self,
        model_spec: Union[YOLOv5_ModelSpec, YOLOv5_TFLite_ModelSpec]
    ):
        super().__init__(model_spec)

        if model_spec.class_names is not None:
            if isinstance(model_spec.class_names, str) or isinstance(model_spec.class_names, Path):
                with fsspec.open(model_spec.class_names, 'r', encoding='utf-8') as out:
                    self.class_names = np.array(json.load(out))
            else:
                self.class_names = np.array(model_spec.class_names)
        else:
            self.class_names = None

        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(
                script_file=model_spec.preprocess_input
            )
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input
        if isinstance(model_spec, YOLOv5_ModelSpec):
            self._load_yolov5_model(model_spec)
            self._raw_predict_images = self._raw_predict_images_torch
        elif isinstance(model_spec, YOLOv5_TFLite_ModelSpec):
            self._load_yolov5_tflite(model_spec)
            self._raw_predict_images = self._raw_predict_images_tflite
        else:
            raise ValueError(
                f"ObjectDetectionAPI_Model got unknown DetectionModelSpec: {type(model_spec)}"
            )

    def _load_yolov5_tflite(self, model_spec: YOLOv5_TFLite_ModelSpec):
        import tensorflow as tf
        temp_file = tempfile.NamedTemporaryFile()
        with fsspec.open(model_spec.model_path, 'rb') as src:
            temp_file.write(src.read())
        model_path = Path(temp_file.name)

        self.model = tf.lite.Interpreter(
            model_path=str(model_path)
        )
        self.model.allocate_tensors()
        self.input_detail = self.model.get_input_details()[0]
        self.output_detail = self.model.get_output_details()[0]
        self.input_dtype = self.input_detail['dtype']
        temp_file.close()

    def _raw_predict_images_torch(
        self,
        input: DetectionInput,
        score_threshold: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        self.model.conf = score_threshold
        results = self.model(input)
        results_pd = results.pandas()

        n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes = [], [], [], []
        for result_pd in results_pd.xyxyn:
            n_raw_bboxes.append(np.array(result_pd[['xmin', 'ymin', 'xmax', 'ymax']]))
            n_raw_keypoints.append(np.array([]).reshape(len(result_pd), 0, 2))
            n_raw_scores.append(np.array(result_pd["confidence"]))
            n_raw_classes.append(np.array(result_pd["class"]))
        return n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes

    def _post_process_raw_predictions_yolov5(self, raw_pred, score_threshold) -> 'CombinedNonMaxSuppression':
        import tensorflow as tf

        def _xywh2xyxy_tf(xywh):
            x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
            return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)

        boxes = _xywh2xyxy_tf(raw_pred[..., :4])
        probs = raw_pred[:, :, 4:5]
        classes = raw_pred[:, :, 5:]
        scores = probs * classes

        boxes = tf.expand_dims(boxes, 2)
        nms = tf.image.combined_non_max_suppression(
            boxes, scores,
            max_output_size_per_class=2000, max_total_size=2000, iou_threshold=0.45,
            score_threshold=score_threshold, clip_boxes=False
        )
        return nms

    def _resize_with_pad(
        self,
        image: np.ndarray,
        target_width: int,
        target_height: int
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        # Resize and pad image while meeting stride-multiple constraints
        # Taken from https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        height, width = image.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        ratio = min(target_height / height, target_width / width)
        # Compute padding
        new_unpad_width, new_unpad_height = int(round(width * ratio)), int(round(height * ratio))
        dw, dh = target_width - new_unpad_width, target_height - new_unpad_height  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if (width, height) != (new_unpad_width, new_unpad_height):  # resize
            image = cv2.resize(image, (new_unpad_width, new_unpad_height), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border

        # Alternative: import tensorflow as tf
        # image = tf.image.resize_with_pad([image], target_height, target_width).numpy()[0]
        # if dh > 0:
        #     image[0:dh//2, :, :] = 114
        #     image[-dh//2:, :, :] = 114
        # if dw > 0:
        #     image[:, :dw//2, :] = 114
        #     image[:, -dw//2:, :] = 114
        return image

    def _scale_coords(
        self,
        img1_size: Tuple[int, int],
        img0_size: Tuple[int, int],
        bboxes: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        # Taken from https://github.com/ultralytics/yolov5/blob/master/utils/general.py
        # Rescale coords (xyxy) from img1_shape to img0_shape
        width1, height1 = img1_size
        width0, height0 = img0_size
        gain = min(height1 / height0, width1 / width0)  # gain  = old / new
        pad = (width1 - width0 * gain) / 2, (height1 - height0 * gain) / 2  # wh padding
        bboxes[:, [0, 2]] -= pad[0]  # x padding
        bboxes[:, [1, 3]] -= pad[1]  # y padding
        bboxes[:, :4] /= gain
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, width0)  # x1, x2
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, height0)  # y1, y2
        return bboxes

    def _raw_predict_images_tflite(
        self,
        input: DetectionInput,
        score_threshold: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        model_width, model_height = self.input_size
        n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes = [], [], [], []
        for image in input:
            height, width = image.shape[:2]
            image = image.astype(np.float32)
            image = self._resize_with_pad(image, target_width=model_width, target_height=model_height)
            image = image[None]
            image /= 255
            int8 = self.input_dtype == np.uint8
            if int8:
                scale, zero_point = self.input_detail['quantization']
                image = (image / scale + zero_point).astype(np.uint8)  # de-scale
            self.model.set_tensor(self.input_detail['index'], image)
            self.model.invoke()
            y = self.model.get_tensor(self.output_detail['index'])
            if int8:
                scale, zero_point = self.output_detail['quantization']
                y = (y.astype(np.float32) - zero_point) * scale  # re-scale
            nms_res = self._post_process_raw_predictions_yolov5(y, score_threshold)
            raw_bboxes = np.array(nms_res.nmsed_boxes[0])  # (ymin, xmin, ymax, xmax)
            raw_scores = np.array(nms_res.nmsed_scores[0])
            nonzero_idxs = (raw_scores > 0)
            raw_bboxes = raw_bboxes[nonzero_idxs]
            raw_bboxes[:, [0, 2]] *= model_width
            raw_bboxes[:, [1, 3]] *= model_height
            raw_bboxes = self._scale_coords((model_width, model_height), (width, height), raw_bboxes)
            raw_bboxes[:, [0, 2]] /= width
            raw_bboxes[:, [1, 3]] /= height

            n_raw_bboxes.append(raw_bboxes)
            n_raw_keypoints.append(np.array([]).reshape(len(raw_bboxes), 0, 2))
            n_raw_scores.append(raw_scores[nonzero_idxs])
            n_raw_classes.append(np.array(nms_res.nmsed_classes[0][nonzero_idxs]))

        return n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes

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
            if (
                xmax - xmin > 0 and ymax - ymin > 0 and
                (xmin, ymin, xmax, ymax) not in bboxes_set
            ):
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
                for class_name in self.class_names[classes.astype(np.int32)]
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

    def predict(
        self,
        input: DetectionInput,
        score_threshold: float,
        classification_top_n: int = 1
    ) -> DetectionOutput:
        heights_widths = [image.shape[:2] for image in input]
        input = self.preprocess_input(input)
        n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes = self._raw_predict_images(input, score_threshold)
        results = [
            self._postprocess_prediction(
                raw_bboxes=raw_bboxes,
                raw_keypoints=raw_keypoints,
                raw_scores=raw_scores,
                raw_classes=raw_classes,
                score_threshold=score_threshold,
                width=width,
                height=height,
                classification_top_n=classification_top_n
            )
            for (height, width), raw_bboxes, raw_keypoints, raw_scores, raw_classes in zip(
                heights_widths, n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes
            )
        ]
        n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k = [
            [res[i] for res in results]
            for i in range(5)
        ]
        return n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k

    def preprocess_input(self, input: DetectionInput):
        return self._preprocess_input(input)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size
