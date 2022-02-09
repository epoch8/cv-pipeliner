import json
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Type, Callable
from pathlib import Path

import numpy as np
import fsspec

from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file
from cv_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from cv_pipeliner.utils.images import denormalize_bboxes


@dataclass
class YOLOv5_ModelSpec(DetectionModelSpec):
    # Can be loaded as torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model_path: Union[None, str, Path, 'torch.nn.Module']  # noqa: F821

    class_names: Union[None, List[str]]
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    input_size: Union[Tuple[int, int], List[int]] = (None, None)

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
        model_spec: YOLOv5_ModelSpec
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

        self._load_yolov5_model(model_spec)
        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(
                script_file=model_spec.preprocess_input
            )
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

    def _raw_predict_images(
        self,
        input: DetectionInput,
        score_threshold: float,
        size: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        self.model.conf = score_threshold
        results = self.model(input, size=size)
        results_pd = results.pandas()

        n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes = [], [], [], []
        for result_pd in results_pd.xyxyn:
            n_raw_bboxes.append(np.array(result_pd[['xmin', 'ymin', 'xmax', 'ymax']]))
            n_raw_keypoints.append(np.array([]).reshape(len(result_pd), 0, 2))
            n_raw_scores.append(np.array(result_pd["confidence"]))
            n_raw_classes.append(np.array(result_pd["class"]))
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
            if xmax - xmin > 0 and ymax - ymin > 0 and (xmin, ymin, xmax, ymax) not in bboxes_set:
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
        classification_top_n: int = 1,
        size: int = 640  # Custom resize image
    ) -> DetectionOutput:
        input = self.preprocess_input(input)
        n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes = self._raw_predict_images(input, score_threshold, size)
        results = [
            self._postprocess_prediction(
                raw_bboxes=raw_bboxes,
                raw_keypoints=raw_keypoints,
                raw_scores=raw_scores,
                raw_classes=raw_classes,
                score_threshold=score_threshold,
                width=image.shape[1],
                height=image.shape[0],
                classification_top_n=classification_top_n
            )
            for image, raw_bboxes, raw_keypoints, raw_scores, raw_classes in zip(
                input, n_raw_bboxes, n_raw_keypoints, n_raw_scores, n_raw_classes
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
        return (None, None)
