from typing import Optional, Union, List, Callable, Type, Tuple, Dict
from pathlib import Path
import numpy as np
import json
import fsspec

from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file
from cv_pipeliner.inference_models.detection.core import (
    DetectionInput,
    DetectionModelSpec, 
    DetectionModel,
    DetectionOutput, 
)


class YOLOv8_ModelSpec(DetectionModelSpec):
    model_name: Optional[str] = None
    model_path: Optional[Union[str, Path]] = None # noqa: F821
    class_names: Optional[List[str]] = None
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
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

        # Loading model
        if isinstance(model_spec, YOLOv8_ModelSpec):
            self._load_yolov8_model(model_spec)
            self._raw_predict_images = self._raw_predict_images_torch
        else:
            raise ValueError(f"ObjectDetectionAPI_Model got unknown DetectionModelSpec: {type(model_spec)}")

    def _load_yolov8_model(self, model_spec: YOLOv8_ModelSpec):
        """YOlOv model initialization

        Args:
            model_spec (YOLOv8_ModelSpec): YOLOv8 Model specification

        Raises:
            ValueError: If model_name and model_path is not specified
        """
        if model_spec.model_name is None and model_spec.model_path is None:
            raise ValueError('Please, specify model name or weights path for loading model')
        
        from ultralytics import YOLO
        
        self.model = YOLO(model_spec.model_path if model_spec.model_path is not None else model_spec.model_name)
        
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
            # TODO how to pass iou threshold?
            iou=0.5
        )

        raw_boxes, raw_keypoints, raw_scores, raw_labels = [], [], [], []
        for prediction in predictions:
            raw_boxes.append(prediction.boxes.xyxy.data.cpu().numpy())
            raw_keypoints.append(np.array([]).reshape(raw_boxes[-1].shape[0], 0, 2))
            raw_labels.append(prediction.boxes.cls.data.cpu().numpy())
            raw_scores.append(prediction.boxes.conf.data.cpu().numpy())

        return raw_boxes, raw_keypoints, raw_scores, raw_labels

    def predict(self, input: DetectionInput, score_threshold: float, classification_top_n: int = None) -> DetectionOutput:
        """Method to run model inference

        Args:
            input (DetectionInput): list of images
            score_threshold (float): model confidence threshold
            classification_top_n (int, optional): .... Defaults to None.

        Returns:
            DetectionOutput: List of boxes, keypoints, scores, classes
        """
        raw_bboxes, raw_keypoints, raw_scores, raw_classes = self._raw_predict_images(input, score_threshold)

        results = (
            [image_boxes.tolist() for image_boxes in raw_bboxes], 
            [image_keypoints.tolist() for image_keypoints in raw_keypoints], 
            [image_scores.tolist() for image_scores in raw_scores], 
            [image_classes.tolist() for image_classes in raw_classes], 
            None
        )

        return results

    def preprocess_input(self, input: DetectionInput) -> DetectionInput:
        self._preprocess_input(input)

    @property
    def input_size(self) -> int:
        return -1