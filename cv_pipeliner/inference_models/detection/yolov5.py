import json
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type, Union

import fsspec
import numpy as np

from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file
from cv_pipeliner.inference_models.detection.core import (
    DetectionInput,
    DetectionModel,
    DetectionModelSpec,
    DetectionOutput,
)
from cv_pipeliner.utils.images import (
    denormalize_bboxes,
    rescale_bboxes_with_pad,
    tf_resize_with_pad,
)


class YOLOv5_ModelSpec(DetectionModelSpec):
    """
    note: model_path can be set as torch.hub.load('ultralytics/yolov5', 'yolov5s')
    """

    model_path: Optional[Union[str, Path, "torch.nn.Module"]]  # noqa: F821
    class_names: Optional[List[str]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    device: str = None
    force_reload: bool = False
    skip_validation: bool = False

    @property
    def inference_model_cls(self) -> Type["YOLOv5_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.yolov5 import YOLOv5_DetectionModel

        return YOLOv5_DetectionModel


class YOLOv5_TFLite_ModelSpec(DetectionModelSpec):
    """
    note: model_path can be set as torch.hub.load('ultralytics/yolov5', 'yolov5s')
    """

    model_path: Union[str, Path]
    bboxes_output_index: Union[int, str]
    scores_output_index: Union[int, str]
    classes_output_index: Union[int, str]
    class_names: Optional[List[str]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    use_default_preprocces_and_postprocess_input: bool = False  # (taken from YOLOv5)

    @property
    def inference_model_cls(self) -> Type["YOLOv5_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.yolov5 import YOLOv5_DetectionModel

        return YOLOv5_DetectionModel


class YOLOv5_TFLiteWithNMS_ModelSpec(DetectionModelSpec):
    """
    note: model_path can be set as torch.hub.load('ultralytics/yolov5', 'yolov5s')
    """

    model_path: Union[str, Path]
    class_names: Optional[List[str]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    use_default_preprocces_and_postprocess_input: bool = False  # (taken from YOLOv5)

    @property
    def inference_model_cls(self) -> Type["YOLOv5_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.yolov5 import YOLOv5_DetectionModel

        return YOLOv5_DetectionModel


class YOLOv5_DetectionModel(DetectionModel):
    def _load_yolov5_model(self, model_spec: YOLOv5_ModelSpec):
        import torch

        if isinstance(model_spec.model_path, torch.nn.Module):
            self.model = model_spec.model_path
            return

        temp_file = tempfile.NamedTemporaryFile(suffix=".pt")
        with fsspec.open(model_spec.model_path, "rb") as src:
            temp_file.write(src.read())
        model_path_tmp = Path(temp_file.name)
        self.model = torch.hub.load(
            "ultralytics/yolov5:v7.0",
            "custom",
            path=str(model_path_tmp),
            force_reload=model_spec.force_reload,
            skip_validation=model_spec.skip_validation,
        )
        if model_spec.device is not None:
            self.model = self.model.to(model_spec.device)
        temp_file.close()

    def __init__(self, model_spec: Union[YOLOv5_ModelSpec, YOLOv5_TFLite_ModelSpec, YOLOv5_TFLiteWithNMS_ModelSpec]):
        super().__init__(model_spec)

        if model_spec.class_names is not None:
            if isinstance(model_spec.class_names, str) or isinstance(model_spec.class_names, Path):
                with fsspec.open(model_spec.class_names, "r", encoding="utf-8") as out:
                    self.class_names = np.array(json.load(out))
            else:
                self.class_names = np.array(model_spec.class_names)
        else:
            self.class_names = None

        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(script_file=model_spec.preprocess_input)
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

        if isinstance(model_spec, YOLOv5_ModelSpec):
            self._load_yolov5_model(model_spec)
            self._raw_predict_images = self._raw_predict_images_torch
        elif isinstance(model_spec, (YOLOv5_TFLite_ModelSpec, YOLOv5_TFLiteWithNMS_ModelSpec)):
            self._load_yolov5_tflite(model_spec)
            if model_spec.use_default_preprocces_and_postprocess_input:
                _, height, width, _ = self.model.get_input_details()[0]["shape"]  # (1 x H x W x 3)
                assert model_spec.preprocess_input is None
                if isinstance(model_spec, YOLOv5_TFLiteWithNMS_ModelSpec):
                    self._preprocess_input = lambda images: [
                        np.array(
                            tf_resize_with_pad(
                                image=image, target_width=width, target_height=height, constant_values=114
                            )
                        )
                        / 255.0
                        for image in images
                    ]
                else:
                    self._preprocess_input = lambda images: [
                        np.array(
                            tf_resize_with_pad(
                                image=image, target_width=width, target_height=height, constant_values=114
                            )
                        )
                        for image in images
                    ]
            self._raw_predict_images = self._raw_predict_images_tflite
        else:
            raise ValueError(f"ObjectDetectionAPI_Model got unknown DetectionModelSpec: {type(model_spec)}")

    def _load_yolov5_tflite(self, model_spec: Union[YOLOv5_TFLite_ModelSpec, YOLOv5_TFLiteWithNMS_ModelSpec]):
        import tensorflow as tf

        temp_file = tempfile.NamedTemporaryFile()
        with fsspec.open(model_spec.model_path, "rb") as src:
            temp_file.write(src.read())
        model_path = Path(temp_file.name)

        self.model = tf.lite.Interpreter(model_path=str(model_path))
        self.model.allocate_tensors()
        self.input_detail = self.model.get_input_details()[0]
        self.input_index = self.input_detail["index"]
        self.input_dtype = self.input_detail["dtype"]
        output_details = self.model.get_output_details()
        output_name_to_index = {output_detail["name"]: output_detail["index"] for output_detail in output_details}
        if isinstance(model_spec, YOLOv5_TFLite_ModelSpec):
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
        elif isinstance(model_spec, YOLOv5_TFLiteWithNMS_ModelSpec):
            self.output_detail = self.model.get_output_details()[0]["index"]
        temp_file.close()

    def _raw_predict_images_torch(
        self, input: DetectionInput, score_threshold: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        self.model.conf = score_threshold
        results = self.model(input)
        results_pd = results.pandas()

        n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes = [], [], [], []
        for result_pd in results_pd.xyxyn:
            n_raw_bboxes.append(np.array(result_pd[["xmin", "ymin", "xmax", "ymax"]]))
            n_raw_keypoints.append(np.array([]).reshape(len(result_pd), 0, 2))
            n_raw_scores.append(np.array(result_pd["confidence"]))
            n_raw_classes.append(np.array(result_pd["class"]))
        return n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes

    def _xywh2xyxy_tf(self, xywh: np.ndarray):
        import tensorflow as tf

        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)

    def _post_process_raw_predictions_yolov5(
        self, raw_preds: np.ndarray, max_output_size: int
    ) -> "CombinedNonMaxSuppression":
        import tensorflow as tf

        bboxes = self._xywh2xyxy_tf(raw_preds[..., :4])
        probs = raw_preds[0, :, 4:5]
        classes = raw_preds[0, :, 5:]
        scores = probs * classes
        weights_count = len(classes)
        bboxes = tf.reshape(bboxes, (weights_count, 4))
        scores = tf.reshape(scores, (weights_count,))
        classes = tf.reshape(classes, (weights_count,))

        selected_indices = tf.image.non_max_suppression(
            bboxes,
            scores,
            max_output_size=max_output_size,
            iou_threshold=0.45,
        )
        output_size = len(selected_indices)
        raw_bboxes = tf.gather(bboxes, selected_indices)
        raw_scores = tf.gather(scores, selected_indices)
        raw_classes = tf.gather(classes, selected_indices)

        raw_bboxes = tf.reshape(raw_bboxes, (1, output_size, 4))
        raw_scores = tf.reshape(raw_scores, (1, output_size))
        raw_classes = tf.reshape(raw_classes, (1, output_size))

        raw_bboxes_np = raw_bboxes.numpy()[0]
        raw_scores_np = raw_scores.numpy()[0]
        raw_classes_np = raw_classes.numpy()[0]

        return raw_bboxes_np, raw_scores_np, raw_classes_np

    def _raw_predict_images_tflite(
        self,
        input: DetectionInput,
        score_threshold: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes = [], [], [], []
        for image in input:
            height, width, _ = image.shape
            image = np.array(image[None, ...], dtype=self.input_dtype)
            self.model.resize_tensor_input(0, [1, height, width, 3])
            self.model.allocate_tensors()
            self.model.set_tensor(self.input_index, image)
            self.model.invoke()
            if isinstance(self.model_spec, YOLOv5_TFLite_ModelSpec):
                raw_bboxes = np.array(self.model.get_tensor(self.bboxes_index))[0]
                raw_scores = np.array(self.model.get_tensor(self.scores_index))[0]
                raw_classes = np.array(self.model.get_tensor(self.classes_index))[0]
            else:
                raw_preds = np.array(self.model.get_tensor(self.output_detail))
                raw_bboxes, raw_scores, raw_classes = self._post_process_raw_predictions_yolov5(
                    raw_preds, max_output_size=2000
                )
            n_raw_bboxes.append(raw_bboxes)
            n_raw_keypoints.append(np.array([]).reshape(len(raw_bboxes), 0, 2))
            n_raw_scores.append(raw_scores)
            n_raw_classes.append(raw_classes)

        return n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes

    def _postprocess_prediction(
        self,
        raw_bboxes: np.ndarray,
        raw_keypoints: np.ndarray,
        raw_scores: np.ndarray,
        raw_classes: np.ndarray,
        score_threshold: float,
        current_width: int,
        current_height: int,
        target_width: int,
        target_height: int,
        classification_top_n: int,
    ) -> Tuple[
        List[Tuple[int, int, int, int]], List[List[Tuple[int, int]]], List[float], List[List[str]], List[List[float]]
    ]:
        if isinstance(self.model_spec, (YOLOv5_TFLite_ModelSpec, YOLOv5_TFLiteWithNMS_ModelSpec)) and (
            self.model_spec.use_default_preprocces_and_postprocess_input
        ):
            # Rescale bboxes
            raw_bboxes = denormalize_bboxes(raw_bboxes, current_width, current_height)
            raw_bboxes = rescale_bboxes_with_pad(
                bboxes=raw_bboxes,
                current_width=current_width,
                current_height=current_height,
                target_width=target_width,
                target_height=target_height,
            )
        else:
            raw_bboxes = denormalize_bboxes(raw_bboxes, target_width, target_height)

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
            if classification_top_n > 1:
                raise NotImplementedError("Not impelemented for classification_top_n > 1")
            class_names_top_n = np.array(
                [
                    [class_name for i in range(classification_top_n)]
                    for class_name in self.class_names[classes.astype(np.int32)]
                ]
            )
            classes_scores_top_n = np.array(
                [[[score for _ in range(classification_top_n)]] for score in classes_scores]
            )
        else:
            class_names_top_n = np.array([[None for _ in range(classification_top_n)] for _ in classes])
            classes_scores_top_n = np.array([[score for _ in range(classification_top_n)] for score in classes_scores])
        return bboxes, keypoints, scores, class_names_top_n, classes_scores_top_n

    def predict(self, input: DetectionInput, score_threshold: float, classification_top_n: int = 1) -> DetectionOutput:
        target_heights_widths = [image.shape[:2] for image in input]
        input = self.preprocess_input(input)
        current_heights_widths = [image.shape[:2] for image in input]
        n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes = self._raw_predict_images(input, score_threshold)
        results = [
            self._postprocess_prediction(
                raw_bboxes=raw_bboxes,
                raw_keypoints=raw_keypoints,
                raw_scores=raw_scores,
                raw_classes=raw_classes,
                score_threshold=score_threshold,
                current_height=current_height,
                current_width=current_width,
                target_width=target_width,
                target_height=target_height,
                classification_top_n=classification_top_n,
            )
            for (
                (target_height, target_width),
                (current_height, current_width),
                raw_bboxes,
                raw_keypoints,
                raw_scores,
                raw_classes,
            ) in zip(
                target_heights_widths,
                current_heights_widths,
                n_raw_bboxes,
                n_raw_keypoints,
                n_raw_scores,
                n_raw_classes,
            )
        ]
        n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k = [
            [res[i] for res in results] for i in range(5)
        ]
        return n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k

    def preprocess_input(self, input: DetectionInput):
        return self._preprocess_input(input)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size
