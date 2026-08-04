import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import fsspec
import numpy as np

from cv_pipeliner.inferencers.backends.preprocess import get_preprocess_input_from_script_file
from cv_pipeliner.inferencers.detection.core import (
    DetectionInput,
    DetectionRuntime,
    DetectionModelSpec,
)
from cv_pipeliner.inferencers.results import DetectionResult, RawDetectionPredictions


class YOLOv8_ModelSpec(DetectionModelSpec):
    model_path: Optional[Union[str, Path]] = None  # noqa: F821
    class_names: Optional[Union[List[str], str, Path]] = None
    keypoints_class_names: Optional[Union[List[str], str, Path]] = None
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    device: str = None
    force_reload: bool = False

    @property
    def runtime_cls(self) -> Type["YOLOv8Runtime"]:
        from cv_pipeliner.inferencers.detection.yolov8 import YOLOv8Runtime

        return YOLOv8Runtime


def _load_names_list(names: Optional[Union[List[str], str, Path]]) -> Optional[np.ndarray]:
    if names is None:
        return None
    if isinstance(names, str) or isinstance(names, Path):
        with fsspec.open(names, "r", encoding="utf-8") as out:
            loaded = np.array(json.load(out))
    else:
        loaded = np.array(names)
    if len(loaded) == 0:
        return None
    return loaded


class YOLOv8Runtime(DetectionRuntime):
    def __init__(self, model_spec: YOLOv8_ModelSpec):
        """YOLOv8 model initialization

        Args:
            model_spec (YOLOv8_ModelSpec): YOLOv8 Model specification

        Raises:
            ValueError: if passed wrong data type of model_spec
        """
        super().__init__(model_spec)

        # Loading classes names and save as attribute
        self.class_names = _load_names_list(model_spec.class_names)
        self.keypoints_class_names = _load_names_list(model_spec.keypoints_class_names)

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
            if self.class_names is None and hasattr(self.model, "names"):
                self.class_names = self._get_class_names_from_model()
            self._raw_predict_images = self._raw_predict_images_torch
        else:
            raise ValueError(f"YOLOv8Runtime got unknown DetectionModelSpec: {type(model_spec)}")

    def _get_class_names_from_model(self) -> np.ndarray:
        model_names = self.model.names
        if isinstance(model_names, dict):
            return np.array([model_names[idx] for idx in sorted(model_names)])
        return np.array(model_names)

    def _build_keypoints_labels(
        self, keypoints_batch: List[Any]
    ) -> Optional[List[List[Optional[List[str]]]]]:
        # Optional: without keypoints_class_names, leave BboxData.keypoints_labels as None.
        if self.keypoints_class_names is None:
            return None
        names = [str(name) for name in self.keypoints_class_names.tolist()]
        n_names = len(names)
        result: List[List[Optional[List[str]]]] = []
        for image_keypoints in keypoints_batch:
            image_labels: List[Optional[List[str]]] = []
            for keypoints in image_keypoints:
                n_keypoints = len(keypoints)
                if n_keypoints == 0:
                    image_labels.append(None)
                elif n_keypoints != n_names:
                    raise ValueError(
                        f"keypoints_class_names length ({n_names}) != keypoints length ({n_keypoints})"
                    )
                else:
                    image_labels.append(list(names))
            result.append(image_labels)
        return result

    def _load_yolov8_model(self, model_spec: YOLOv8_ModelSpec):
        """YOLOv8 model initialization

        Args:
            model_spec (YOLOv8_ModelSpec): YOLOv8 Model specification

        Raises:
            ValueError: If model_path is not specified
        """
        if model_spec.model_path is None:
            raise ValueError("Please, specify model_path for loading model")

        from ultralytics import YOLO

        model_path = Path(model_spec.model_path) if not isinstance(model_spec.model_path, Path) else model_spec.model_path
        model_path_str = str(model_spec.model_path)

        if model_path.exists():
            self.model = YOLO(model_path)
        elif "://" in model_path_str:
            temp_file = tempfile.NamedTemporaryFile(suffix=".pt")
            with fsspec.open(model_path_str, "rb") as src:
                temp_file.write(src.read())
            self.model = YOLO(Path(temp_file.name))
        else:
            # Ultralytics hub name (e.g. yolov8n.pt) or other non-local identifier
            self.model = YOLO(model_path_str)

        if model_spec.device is not None:
            self.model = self.model.to(model_spec.device)

    def _raw_predict_images_torch(self, input: DetectionInput, score_threshold: float) -> RawDetectionPredictions:
        predictions = self.model.predict(
            input,
            verbose=False,
            save_conf=True,
            conf=score_threshold,
            # retina_masks=True,
        )
        bboxes, keypoints, keypoints_scores, masks, scores, class_ids = [], [], [], [], [], []
        for prediction in predictions:
            bboxes.append(prediction.boxes.xyxy.data.cpu().numpy())
            if prediction.keypoints is not None:
                keypoints.append(prediction.keypoints.xy.data.cpu().numpy())
                if getattr(prediction.keypoints, "conf", None) is not None:
                    keypoints_scores.append(prediction.keypoints.conf.data.cpu().numpy())
                else:
                    keypoints_scores.append(None)
            else:
                keypoints.append(np.array([]).reshape(len(bboxes[-1]), 0, 2))
                keypoints_scores.append(None)
            if prediction.masks is not None:
                all_polygons = prediction.masks.xy
                masks.append([[polygon] for polygon in all_polygons])
            else:
                masks.append([[[]] for _ in range(len(bboxes[-1]))])
            class_ids.append(prediction.boxes.cls.data.cpu().numpy())
            scores.append(prediction.boxes.conf.data.cpu().numpy())

        return RawDetectionPredictions(
            bboxes=bboxes,
            keypoints=keypoints,
            detection_scores=scores,
            class_ids=class_ids,
            masks=masks,
            keypoints_scores=keypoints_scores,
        )

    def predict(
        self,
        input: DetectionInput,
        score_threshold: float,
        classification_top_n: int = 1,
    ) -> DetectionResult:
        input = self._preprocess_input(input)
        # Rebatch by image size due to https://github.com/ultralytics/ultralytics/issues/15430
        size_to_images: Dict[Tuple[int, ...], List[np.ndarray]] = {}
        size_to_idxs: Dict[Tuple[int, ...], List[int]] = {}
        for idx, image in enumerate(input):
            size = tuple(image.shape[0:3])
            size_to_images.setdefault(size, []).append(image)
            size_to_idxs.setdefault(size, []).append(idx)

        idx_to_raw = {}
        for size, images in size_to_images.items():
            raw_batch = self._raw_predict_images(images, score_threshold)
            for i, idx in enumerate(size_to_idxs[size]):
                idx_to_raw[idx] = raw_batch[i]
        raw = RawDetectionPredictions.from_images([idx_to_raw[idx] for idx in range(len(input))])

        if self.class_names is not None:
            if classification_top_n > 1:
                raise NotImplementedError("Not impelemented for classification_top_n > 1")
            labels_top_n = [
                [
                    [class_name for _ in range(classification_top_n)]
                    for class_name in self.class_names[classes.astype(np.int32)]
                ]
                for classes in raw.class_ids
            ]
            classification_scores_top_n = [[[score] for score in scores] for scores in raw.detection_scores]
        else:
            labels_top_n = [[None for _ in range(classification_top_n)] for _ in raw.class_ids]
            classification_scores_top_n = [[score for _ in range(classification_top_n)] for score in raw.detection_scores]

        keypoints_scores: List[List[Optional[List[float]]]] = []
        for image_keypoints, image_scores in zip(raw.keypoints, raw.keypoints_scores or [None] * len(raw)):
            if image_scores is None:
                keypoints_scores.append([None for _ in range(len(image_keypoints))])
            else:
                keypoints_scores.append(np.asarray(image_scores).astype(float).tolist())

        keypoints_labels = self._build_keypoints_labels(raw.keypoints)

        return DetectionResult(
            bboxes=[image_boxes.tolist() for image_boxes in raw.bboxes],
            keypoints=[np.array(image_keypoints).round().astype(np.int32).tolist() for image_keypoints in raw.keypoints],
            masks=[
                [
                    [np.array(polygon).round().astype(np.int32).tolist() for polygon in polygons]
                    for polygons in image_masks
                ]
                for image_masks in (raw.masks or [])
            ],
            detection_scores=[image_scores.tolist() for image_scores in raw.detection_scores],
            labels_top_n=labels_top_n,
            classification_scores_top_n=classification_scores_top_n,
            keypoints_scores=keypoints_scores,
            keypoints_labels=keypoints_labels,
        )

    def preprocess_input(self, input: DetectionInput) -> DetectionInput:
        return self._preprocess_input(input)

    @property
    def input_size(self) -> int:
        return self.model.imgsz
