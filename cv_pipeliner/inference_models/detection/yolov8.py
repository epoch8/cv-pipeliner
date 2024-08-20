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


class YOLOv8_ModelSpec(DetectionModelSpec):
    model_name: Optional[str] = None
    model_path: Optional[Union[str, Path]] = None  # noqa: F821
    class_names: Optional[Union[List[str], str, Path]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    device: str = None
    force_reload: bool = False

    @property
    def inference_model_cls(self) -> Type["YOLOv8_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.yolov8 import YOLOv8_DetectionModel

        return YOLOv8_DetectionModel


class YOLOv8_DetectionModel(DetectionModel):
    def __init__(self, model_spec: YOLOv8_ModelSpec):
        """YOLOv8 model initialization

        Args:
            model_spec (YOLOv8_ModelSpec): YOLOv8 Model specification

        Raises:
            ValueError: if passed wrong data type of model_spec
        """
        super().__init__(model_spec)

        # Loading classes names and save as attribute
        if model_spec.class_names is not None:
            if isinstance(model_spec.class_names, str) or isinstance(model_spec.class_names, Path):
                with fsspec.open(model_spec.class_names, "r", encoding="utf-8") as out:
                    self.class_names = np.array(json.load(out))
            else:
                self.class_names = np.array(model_spec.class_names)
        else:
            self.class_names = None

        # Loading preprocessing function
        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(script_file=model_spec.preprocess_input)
        else:
            if model_spec.preprocess_input is None:
                # For numpy.ndarray inputs, YOLOv8 expects BGR format
                # https://github.com/ultralytics/ultralytics/issues/2575
                self._preprocess_input = lambda input: [image[:, :, ::-1] for image in input]
            else:
                self._preprocess_input = model_spec.preprocess_input

        # Loading model
        if isinstance(model_spec, YOLOv8_ModelSpec):
            self._load_yolov8_model(model_spec)
            self._raw_predict_images = self._raw_predict_images_torch
        else:
            raise ValueError(f"ObjectDetectionAPI_Model got unknown DetectionModelSpec: {type(model_spec)}")

    def _load_yolov8_model(self, model_spec: YOLOv8_ModelSpec):
        """YOLOv8 model initialization

        Args:
            model_spec (YOLOv8_ModelSpec): YOLOv8 Model specification

        Raises:
            ValueError: If model_name and model_path is not specified
        """
        if model_spec.model_name is None and model_spec.model_path is None:
            raise ValueError("Please, specify model name or weights path for loading model")

        if model_spec.model_path is not None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".pt")
            with fsspec.open(model_spec.model_path, "rb") as src:
                temp_file.write(src.read())
            model_path_tmp = Path(temp_file.name)
            from ultralytics import YOLO

            self.model = YOLO(model_path_tmp)
        else:
            self.model = YOLO(model_spec.model_name)

        if model_spec.device is not None:
            self.model = self.model.to(model_spec.device)

    def _raw_predict_images_torch(
        self, input: DetectionInput, score_threshold: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:
        """Private method to run pytorch model inference and return raw results

        Args:
            input (DetectionInput): list of images
            score_threshold (float): model confidence threshold

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]: _description_
        """

        predictions = self.model.predict(
            input,
            verbose=False,
            save_conf=True,
            conf=score_threshold,
            retina_masks=True,
        )
        raw_boxes, raw_keypoints, raw_masks, raw_scores, raw_labels = [], [], [], [], []
        for prediction in predictions:
            raw_boxes.append(prediction.boxes.xyxy.data.cpu().numpy())
            if prediction.masks is not None:
                all_keypoints = prediction.masks.xy
                raw_masks.append([[kp] for kp in all_keypoints])
            else:
                raw_masks.append([[[]] for _ in range(len(raw_boxes[-1])) for _ in range(len(raw_boxes))])
            raw_keypoints.append(np.array([]).reshape(len(raw_boxes[-1]), 0, 2))  # TODO: add keypoints support
            raw_labels.append(prediction.boxes.cls.data.cpu().numpy())
            raw_scores.append(prediction.boxes.conf.data.cpu().numpy())

        return raw_boxes, raw_keypoints, raw_masks, raw_scores, raw_labels

    def predict(
        self,
        input: DetectionInput,
        score_threshold: float,
        classification_top_n: int = 1,
    ) -> DetectionOutput:
        """Method to run model inference

        Args:
            input (DetectionInput): list of images
            score_threshold (float): model confidence threshold
            classification_top_n (int, optional): .... Defaults to None.

        Returns:
            DetectionOutput: List of boxes, keypoints, scores, classes
        """
        input = self._preprocess_input(input)
        # Try to rebatch to batches of one size due to https://github.com/ultralytics/ultralytics/issues/15430
        size_to_images = {}
        size_to_idxs = {}
        for idx, image in enumerate(input):
            size = tuple(image.shape[0:3])
            if size not in size_to_images:
                size_to_images[size] = []
                size_to_idxs[size] = []
            size_to_images[size].append(image)
            size_to_idxs[size].append(idx)

        idx_to_results = {}
        for size, images in size_to_images.items():
            (
                size_raw_bboxes,
                size_raw_keypoints,
                size_raw_masks,
                size_raw_scores,
                size_raw_classes,
            ) = self._raw_predict_images(images, score_threshold)
            for i, idx in enumerate(size_to_idxs[size]):
                idx_to_results[idx] = (
                    size_raw_bboxes[i],
                    size_raw_keypoints[i],
                    size_raw_masks[i],
                    size_raw_scores[i],
                    size_raw_classes[i],
                )
        results = [idx_to_results[idx] for idx in range(len(input))]
        raw_bboxes, raw_keypoints, raw_masks, raw_scores, raw_classes = zip(*results)

        if self.class_names is not None:
            if classification_top_n > 1:
                raise NotImplementedError("Not impelemented for classification_top_n > 1")
            class_names_top_n = [
                [
                    [class_name for i in range(classification_top_n)]
                    for class_name in self.class_names[classes.astype(np.int32)]
                ]
                for classes in raw_classes
            ]
            classes_scores_top_n = [[[score] for score in scores] for scores in raw_scores]
        else:
            class_names_top_n = [[None for _ in range(classification_top_n)] for _ in raw_classes]
            classes_scores_top_n = [[score for _ in range(classification_top_n)] for score in raw_scores]

        n_pred_bboxes = [image_boxes.tolist() for image_boxes in raw_bboxes]
        n_pred_keypoints = [np.array(k_keypoints).round().astype(np.int32).tolist() for k_keypoints in raw_keypoints]
        n_pred_masks = [
            [
                [np.array(polygon).round().astype(np.int32).tolist() for polygon in k_polygons]
                for k_polygons in n_k_polygons
            ]
            for n_k_polygons in raw_masks
        ]
        n_pred_scores = [image_scores.tolist() for image_scores in raw_scores]
        n_pred_class_names_top_k = class_names_top_n
        n_pred_scores_top_k = classes_scores_top_n
        return (
            n_pred_bboxes,
            n_pred_keypoints,
            n_pred_masks,
            n_pred_scores,
            n_pred_class_names_top_k,
            n_pred_scores_top_k,
        )

    def preprocess_input(self, input: DetectionInput) -> DetectionInput:
        return self._preprocess_input(input)

    @property
    def input_size(self) -> int:
        return self.model.imgsz
