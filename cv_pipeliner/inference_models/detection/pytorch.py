import json
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union, Type, Literal
from pathlib import Path

import numpy as np
import cv2
import fsspec
from pathy import Pathy

from cv_pipeliner.inference_models.detection.core import (
    DetectionModelSpec, DetectionModel, DetectionInput, DetectionOutput
)
from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file


@dataclass
class PytorchDetection_ModelSpec(DetectionModelSpec):
    input_size: Union[Tuple[int, int], List[int]]
    model_path: Union[str, Pathy]  # can be also tf.keras.Model

    bboxes_output_index: int
    scores_output_index: int
    classes_output_index: int
    input_format: Literal['RGB', 'BGR']
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    keypoints_output_index: Union[int, None] = None
    class_names: Union[List[str], str, Path, None] = None
    device: Literal['cpu', 'cuda'] = 'cpu'


    @property
    def inference_model_cls(self) -> Type['Pytorch_DetectionModel']:
        from cv_pipeliner.inference_models.detection.pytorch import Pytorch_DetectionModel
        return Pytorch_DetectionModel


class Pytorch_DetectionModel(DetectionModel):
    def _load_pt_model(
        self,
        model_spec: PytorchDetection_ModelSpec
    ):
        import torch
        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)
        model_config_path = temp_dir_path / Pathy(model_spec.model_path).name
        with open(model_config_path, 'wb') as out:
            with fsspec.open(model_spec.model_path, 'rb') as src:
                out.write(src.read())
        self.model = torch.jit.load(model_config_path).to(model_spec.device)
        self.model.eval()
        temp_dir.cleanup()

    def __init__(
        self,
        model_spec: Union[
            PytorchModelSpec
        ],
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

        if isinstance(model_spec, PytorchModelSpec):
            self._load_pt_model(model_spec)
            self.device = model_spec.device
            self.input_format = model_spec.device
        else:
            raise ValueError(
                f"{Pytorch_DetectionModel.__name__} got unknown DetectionModelSpec: {type(model_spec)}"
            )

        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(
                script_file=model_spec.preprocess_input
            )
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

    def _raw_predict_single_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import torch
        if self.input_format == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = torch.tensor(image).permute(2, 1, 0).to(self.device)
        predictions = self.model(image)

        raw_bboxes = predictions[self.model_spec.bboxes_output_index].detach().cpu().numpy()
        if self.model_spec.keypoints_output_index is None:
            raw_keypoints = np.array([]).reshape(len(raw_bboxes), 0, 2)
        else:
            raw_keypoints = predictions[
                self.model_spec.keypoints_output_index
            ].detach().cpu().numpy()[:, :, :2]
        raw_scores = predictions[self.model_spec.scores_output_index].detach().cpu().numpy()
        raw_classes = predictions[self.model_spec.classes_output_index].detach().cpu().numpy()

        return raw_bboxes, raw_keypoints, raw_scores, raw_classes

    def _postprocess_prediction(
        self,
        raw_bboxes: np.ndarray,
        raw_keypoints: np.ndarray,
        raw_scores: np.ndarray,
        raw_classes: np.ndarray,
        score_threshold: float,
        classification_top_n: int,
        height: int,
        width: int
    ) -> Tuple[
        List[Tuple[int, int, int, int]],
        List[List[Tuple[int, int]]],
        List[float], List[List[str]], List[List[float]]
    ]:
        raw_bboxes[:, [0, 2]] = (raw_bboxes[:, [0, 2]] / self.input_size[0] * width).astype(int)
        raw_bboxes[:, [1, 3]] = (raw_bboxes[:, [1, 3]] / self.input_size[1] * height).astype(int)
        raw_keypoints[:, :, 0] = (raw_keypoints[:, :, 0] / self.input_size[0] * width).astype(int)
        raw_keypoints[:, :, 1] = (raw_keypoints[:, :, 1] / self.input_size[1] * height).astype(int)

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
                for class_name in self.class_names[(classes.astype(np.int32))]
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
        (
            n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k
        ) = [], [], [], [], []

        for image in input:
            height, width, _ = image.shape
            resized_image = self.preprocess_input([image])[0]
            raw_bboxes, raw_keypoints, raw_scores, raw_classes = self._raw_predict_single_image(resized_image)
            bboxes, keypoints, scores, class_names_top_k, classes_scores_top_k = self._postprocess_prediction(
                raw_bboxes=raw_bboxes,
                raw_keypoints=raw_keypoints,
                raw_scores=raw_scores,
                raw_classes=raw_classes,
                score_threshold=score_threshold,
                classification_top_n=classification_top_n,
                width=width,
                height=height
            )

            n_pred_bboxes.append(bboxes)
            n_pred_keypoints.append(keypoints)
            n_pred_scores.append(scores)
            n_pred_class_names_top_k.append(class_names_top_k)
            n_pred_scores_top_k.append(classes_scores_top_k)

        return n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k

    def preprocess_input(self, input: DetectionInput):
        return self._preprocess_input(input)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.model_spec.input_size
