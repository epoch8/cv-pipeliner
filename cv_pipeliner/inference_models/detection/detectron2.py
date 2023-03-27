import json
import tempfile
from typing import Callable, List, Tuple, Union, Type, Literal
from pathlib import Path

import numpy as np
import cv2
import fsspec
from pathy import Pathy

from cv_pipeliner.inference_models.detection.core import (
    DetectionModelSpec,
    DetectionModel,
    DetectionInput,
    DetectionOutput,
)
from cv_pipeliner.core.inference_model import get_preprocess_input_from_script_file


class Detectron2_ModelSpec(DetectionModelSpec):
    input_size: Union[Tuple[int, int], List[int]]
    model_path: Union[str, Pathy]  # can be also tf.keras.Model

    bboxes_output_index: int
    scores_output_index: int
    classes_output_index: int
    input_format: Literal["RGB", "BGR"]
    preprocess_input: Union[Callable[[List[np.ndarray]], np.ndarray], str, Path, None] = None
    keypoints_output_index: Union[int, None] = None
    keypoints_heatmap_index: Union[int, None] = None
    class_names: Union[List[str], str, Path, None] = None
    device: Literal["cpu", "cuda"] = "cpu"
    input_type: Literal["detectron2", "caffe2"] = "detectron2"

    @property
    def inference_model_cls(self) -> Type["Detectron2_DetectionModel"]:
        from cv_pipeliner.inference_models.detection.detectron2 import Detectron2_DetectionModel

        return Detectron2_DetectionModel


def heatmaps_to_keypoints(maps: "torch.Tensor", rois: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
    """
    Extract predicted keypoint locations from heatmaps.
    Args:
        maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W). The predicted heatmap of logits for
            each ROI and each keypoint.
        rois (Tensor): (#ROIs, 4). The box of each ROI.
    Returns:
        Tensor of shape (#ROIs, #keypoints, 4) with the last dimension corresponding to
        (x, y, logit, score) for each keypoint.
    When converting discrete pixel indices in an NxN image to a continuous keypoint coordinate,
    we maintain consistency with :meth:`Keypoints.to_heatmap` by using the conversion from
    Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    """
    import torch
    from torch.nn import functional as F

    # The decorator use of torch.no_grad() was not supported by torchscript.
    # https://github.com/pytorch/pytorch/issues/44768
    maps = maps.detach()
    rois = rois.detach()

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_rois, num_keypoints = maps.shape[:2]
    xy_preds = maps.new_zeros(rois.shape[0], num_keypoints, 4)

    width_corrections = widths / widths_ceil
    height_corrections = heights / heights_ceil

    keypoints_idx = torch.arange(num_keypoints, device=maps.device)

    for i in range(num_rois):
        outsize = (int(heights_ceil[i]), int(widths_ceil[i]))
        roi_map = F.interpolate(maps[[i]], size=outsize, mode="bicubic", align_corners=False).squeeze(
            0
        )  # #keypoints x H x W

        # softmax over the spatial region
        max_score, _ = roi_map.view(num_keypoints, -1).max(1)
        max_score = max_score.view(num_keypoints, 1, 1)
        tmp_full_resolution = (roi_map - max_score).exp_()
        tmp_pool_resolution = (maps[i] - max_score).exp_()
        # Produce scores over the region H x W, but normalize with POOL_H x POOL_W,
        # so that the scores of objects of different absolute sizes will be more comparable
        roi_map_scores = tmp_full_resolution / tmp_pool_resolution.sum((1, 2), keepdim=True)

        w = roi_map.shape[2]
        pos = roi_map.view(num_keypoints, -1).argmax(1)

        x_int = pos % w
        y_int = (pos - x_int) // w

        assert (roi_map_scores[keypoints_idx, y_int, x_int] == roi_map_scores.view(num_keypoints, -1).max(1)[0]).all()

        x = (x_int.float() + 0.5) * width_corrections[i]
        y = (y_int.float() + 0.5) * height_corrections[i]

        xy_preds[i, :, 0] = x + offset_x[i]
        xy_preds[i, :, 1] = y + offset_y[i]
        xy_preds[i, :, 2] = roi_map[keypoints_idx, y_int, x_int]
        xy_preds[i, :, 3] = roi_map_scores[keypoints_idx, y_int, x_int]

    return xy_preds


class Detectron2_DetectionModel(DetectionModel):
    def _load_pt_model(self, model_spec: Detectron2_ModelSpec):
        import torch

        temp_dir = tempfile.TemporaryDirectory()
        temp_dir_path = Path(temp_dir.name)
        model_config_path = temp_dir_path / Pathy(model_spec.model_path).name
        with open(model_config_path, "wb") as out:
            with fsspec.open(model_spec.model_path, "rb") as src:
                out.write(src.read())
        self.model = torch.jit.load(model_config_path).to(model_spec.device)
        self.model.eval()
        temp_dir.cleanup()

    def __init__(self, model_spec: Detectron2_ModelSpec):
        super().__init__(model_spec)

        if model_spec.class_names is not None:
            if isinstance(model_spec.class_names, str) or isinstance(model_spec.class_names, Path):
                with fsspec.open(model_spec.class_names, "r", encoding="utf-8") as out:
                    self.class_names = np.array(json.load(out))
            else:
                self.class_names = np.array(model_spec.class_names)
        else:
            self.class_names = None

        if isinstance(model_spec, Detectron2_ModelSpec):
            self._load_pt_model(model_spec)
            self.device = model_spec.device
            self.input_format = model_spec.device
            self.input_type = model_spec.input_type
        else:
            raise ValueError(f"{Detectron2_DetectionModel.__name__} got unknown DetectionModelSpec: {type(model_spec)}")

        if isinstance(model_spec.preprocess_input, str) or isinstance(model_spec.preprocess_input, Path):
            self._preprocess_input = get_preprocess_input_from_script_file(script_file=model_spec.preprocess_input)
        else:
            if model_spec.preprocess_input is None:
                self._preprocess_input = lambda x: x
            else:
                self._preprocess_input = model_spec.preprocess_input

    def _raw_predict_single_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import torch

        if self.input_format == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = torch.tensor(image).permute(2, 1, 0).to(self.device)

        if self.input_type == "detectron2":
            predictions = self.model(image)
        elif self.input_type == "caffe2":
            image = image[None, ...]
            im_info = torch.tensor([[*self.input_size, 1.0]])
            predictions = self.model((image, im_info))

        raw_bboxes = predictions[self.model_spec.bboxes_output_index]
        if self.model_spec.keypoints_output_index is None:
            if self.model_spec.keypoints_heatmap_index is not None:
                raw_keypoints_heatmaps = predictions[self.model_spec.keypoints_heatmap_index]
                raw_keypoints = heatmaps_to_keypoints(raw_keypoints_heatmaps, raw_bboxes)
            else:
                raw_keypoints = np.array([]).reshape(len(raw_bboxes), 0, 2)
        else:
            raw_keypoints = predictions[self.model_spec.keypoints_output_index]

        raw_bboxes = raw_bboxes.detach().cpu().numpy()
        raw_keypoints = raw_keypoints.detach().cpu().numpy()[:, :, :2]
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
        width: int,
    ) -> Tuple[
        List[Tuple[int, int, int, int]], List[List[Tuple[int, int]]], List[float], List[List[str]], List[List[float]]
    ]:
        raw_bboxes[:, [0, 2]] = raw_bboxes[:, [0, 2]] / self.input_size[0] * width
        raw_bboxes[:, [1, 3]] = raw_bboxes[:, [1, 3]] / self.input_size[1] * height
        raw_bboxes = raw_bboxes.round().astype(int)
        raw_keypoints[:, :, 0] = raw_keypoints[:, :, 0] / self.input_size[0] * width
        raw_keypoints[:, :, 1] = raw_keypoints[:, :, 1] / self.input_size[1] * height
        raw_keypoints = raw_keypoints.round().astype(int)
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
            class_names_top_n = np.array(
                [
                    [class_name for i in range(classification_top_n)]
                    for class_name in self.class_names[(classes.astype(np.int32))]
                ]
            )
            classes_scores_top_n = np.array([[score for _ in range(classification_top_n)] for score in classes_scores])
        else:
            class_names_top_n = np.array([[None for _ in range(classification_top_n)] for _ in classes])
            classes_scores_top_n = np.array([[score for _ in range(classification_top_n)] for score in classes_scores])

        return bboxes, keypoints, scores, class_names_top_n, classes_scores_top_n

    def predict(self, input: DetectionInput, score_threshold: float, classification_top_n: int = 1) -> DetectionOutput:
        (n_pred_bboxes, n_pred_keypoints, n_pred_scores, n_pred_class_names_top_k, n_pred_scores_top_k) = (
            [],
            [],
            [],
            [],
            [],
        )

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
                height=height,
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
