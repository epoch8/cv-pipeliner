"""RF-DETR detection / segmentation runtime for cv-pipeliner.

Detection uses `RFDETR*` constructors; segmentation uses `RFDETRSeg*`. Both return the shared
`DetectionResult` shape (masks empty for detection; polygon masks for segmentation).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Type, Union

import fsspec
import numpy as np

from cv_pipeliner.inferencers.backends.preprocess import get_preprocess_input_from_script_file
from cv_pipeliner.inferencers.detection.core import DetectionInput, DetectionModelSpec, DetectionRuntime
from cv_pipeliner.inferencers.results import DetectionResult

TaskName = Literal["detect", "segment"]

_DET_VARIANTS = ("nano", "small", "medium", "large", "xlarge", "2xlarge", "2xl")
_SEG_VARIANTS = ("nano", "small", "medium", "large", "xlarge", "2xlarge", "2xl")


class RFDETR_ModelSpec(DetectionModelSpec):
    """Model specification for Roboflow RF-DETR (detection or instance segmentation)."""

    model_path: Optional[Union[str, Path]] = None
    class_names: Optional[Union[List[str], str, Path]] = None
    input_size: Optional[Union[int, Tuple[int, int]]] = None
    variant: str = "medium"
    task: TaskName = "detect"
    preprocess_input: Union[Callable[[List[np.ndarray]], List[np.ndarray]], str, Path, None] = None
    device: Optional[str] = None
    optimize_for_inference: bool = False

    @property
    def runtime_cls(self) -> Type["RFDETRRuntime"]:
        from cv_pipeliner.inferencers.detection.rfdetr import RFDETRRuntime

        return RFDETRRuntime


def _load_names_list(names: Optional[Union[List[str], str, Path]]) -> Optional[np.ndarray]:
    if names is None:
        return None
    if isinstance(names, (str, Path)):
        with fsspec.open(names, "r", encoding="utf-8") as out:
            loaded = np.array(json.load(out))
    else:
        loaded = np.array(names)
    if len(loaded) == 0:
        return None
    return loaded


def _normalize_variant(variant: str) -> str:
    value = (variant or "medium").lower().strip()
    if value == "2xl":
        return "2xlarge"
    return value


def _mask_to_polygons(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Convert a binary mask to cv_pipeliner polygon masks (list of polygons as int xy pairs)."""
    import cv2

    mask = np.ascontiguousarray(mask)
    contours_result = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours_result[-2]
    polygons: List[List[Tuple[int, int]]] = []
    for contour in contours:
        flat = contour.reshape(-1, 2)
        if len(flat) < 3:
            continue
        polygons.append([(int(round(x)), int(round(y))) for x, y in flat])
    return polygons if polygons else [[]]


def _empty_keypoints(n_boxes: int) -> List[List[Tuple[int, int]]]:
    return [[] for _ in range(n_boxes)]


class RFDETRRuntime(DetectionRuntime):
    def __init__(self, model_spec: RFDETR_ModelSpec):
        super().__init__(model_spec)
        if not isinstance(model_spec, RFDETR_ModelSpec):
            raise ValueError(f"RFDETRRuntime got unknown DetectionModelSpec: {type(model_spec)}")

        self.class_names = _load_names_list(model_spec.class_names)
        self._task: TaskName = model_spec.task
        self._variant = _normalize_variant(model_spec.variant)
        self._input_size = self._resolve_input_size(model_spec.input_size)

        if isinstance(model_spec.preprocess_input, (str, Path)):
            self._preprocess_input = get_preprocess_input_from_script_file(script_file=model_spec.preprocess_input)
        elif model_spec.preprocess_input is None:
            # ImageData / datapipe paths feed RGB arrays; RF-DETR expects RGB.
            self._preprocess_input = lambda images: list(images)
        else:
            self._preprocess_input = model_spec.preprocess_input

        self.model = self._load_model(model_spec)
        if self.class_names is None:
            self.class_names = self._class_names_from_model()

    @staticmethod
    def _resolve_input_size(input_size: Optional[Union[int, Tuple[int, int]]]) -> Tuple[int, int]:
        if input_size is None:
            return (560, 560)
        if isinstance(input_size, int):
            return (input_size, input_size)
        if len(input_size) != 2:
            raise ValueError(f"input_size must be int or (h, w), got {input_size!r}")
        return (int(input_size[0]), int(input_size[1]))

    def _load_model(self, model_spec: RFDETR_ModelSpec):
        if model_spec.model_path is None:
            raise ValueError("Please, specify model_path for loading RF-DETR model")

        try:
            import rfdetr
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RF-DETR is not installed. Install with `pip install 'cv_pipeliner[rfdetr]'` "
                "or `pip install rfdetr`."
            ) from exc

        def _optional_ctor(name: str):
            # rfdetr.__getattr__ raises ImportError for plus-only symbols (RFDETRXLarge, …)
            # when rfdetr[plus] is not installed — getattr(..., None) does not catch that.
            if name not in rfdetr.__dict__:
                return None
            return rfdetr.__dict__[name]

        if self._task == "segment":
            variant_to_ctor = {
                "nano": rfdetr.RFDETRSegNano,
                "small": rfdetr.RFDETRSegSmall,
                "medium": rfdetr.RFDETRSegMedium,
                "large": rfdetr.RFDETRSegLarge,
                "xlarge": rfdetr.RFDETRSegXLarge,
                "2xlarge": rfdetr.RFDETRSeg2XLarge,
            }
            allowed = _SEG_VARIANTS
        else:
            variant_to_ctor = {
                "nano": rfdetr.RFDETRNano,
                "small": rfdetr.RFDETRSmall,
                "medium": rfdetr.RFDETRMedium,
                "large": rfdetr.RFDETRLarge,
                "xlarge": _optional_ctor("RFDETRXLarge"),
                "2xlarge": _optional_ctor("RFDETR2XLarge"),
            }
            variant_to_ctor = {k: v for k, v in variant_to_ctor.items() if v is not None}
            allowed = tuple(variant_to_ctor)

        if self._variant not in variant_to_ctor:
            raise ValueError(
                f"Unknown RF-DETR variant={model_spec.variant!r} for task={self._task!r}. "
                f"Expected one of: {sorted(allowed)}"
            )

        weights_path = self._materialize_weights(model_spec.model_path)
        ctor_kwargs: dict = {"pretrain_weights": str(weights_path)}
        if self.class_names is not None and len(self.class_names) > 0:
            # Match the fine-tuned head size; RF-DETR pretrained defaults to 90 COCO classes.
            ctor_kwargs["num_classes"] = int(len(self.class_names))
        if self._input_size[0] == self._input_size[1]:
            ctor_kwargs["resolution"] = int(self._input_size[0])
        model = variant_to_ctor[self._variant](**ctor_kwargs)
        if model_spec.optimize_for_inference:
            try:
                model.optimize_for_inference(compile=True, batch_size=1)
            except Exception:
                pass
        return model

    @staticmethod
    def _materialize_weights(model_path: Union[str, Path]) -> Path:
        path = Path(model_path) if not isinstance(model_path, Path) else model_path
        path_str = str(model_path)
        if path.exists():
            return path
        if "://" in path_str:
            suffix = Path(path_str.split("?")[0]).suffix or ".pth"
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            with fsspec.open(path_str, "rb") as src:
                temp_file.write(src.read())
            temp_file.flush()
            temp_file.close()
            return Path(temp_file.name)
        raise FileNotFoundError(f"RF-DETR model_path not found: {model_path!r}")

    def _class_names_from_model(self) -> Optional[np.ndarray]:
        names = getattr(self.model, "class_names", None)
        if names is None:
            return None
        if isinstance(names, dict):
            return np.array([names[idx] for idx in sorted(names)])
        return np.array(list(names))

    def _label_for_class_id(self, cid: int) -> str:
        """Map RF-DETR class_id to a label.

        Datapipe GT→COCO uses 1-based category ids, and fine-tuned RF-DETR typically
        emits those. Prefer 1-based mapping; fall back to 0-based for contiguous heads.
        """
        assert self.class_names is not None
        names = self.class_names
        i = int(cid)
        n = len(names)
        if 1 <= i <= n:
            return str(names[i - 1])
        if 0 <= i < n:
            return str(names[i])
        return str(i)

    def _predict_one(self, image: np.ndarray, score_threshold: float) -> Tuple[
        List[List[int]],
        List[float],
        List[int],
        List[List[List[Tuple[int, int]]]],
    ]:
        detections = self.model.predict(image, threshold=float(score_threshold))
        boxes = getattr(detections, "xyxy", None)
        scores = getattr(detections, "confidence", None)
        class_ids = getattr(detections, "class_id", None)
        masks = getattr(detections, "mask", None)

        if boxes is None or scores is None or class_ids is None:
            raise RuntimeError(
                "Unexpected RF-DETR output. Expected supervision Detections with xyxy/confidence/class_id."
            )

        boxes_arr = np.asarray(boxes)
        scores_arr = np.asarray(scores, dtype=float)
        class_ids_arr = np.asarray(class_ids).astype(int)
        n = len(boxes_arr)

        bboxes = [
            [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
            for x1, y1, x2, y2 in boxes_arr.reshape(n, 4)
        ]
        detection_scores = scores_arr.reshape(n).tolist()
        ids = class_ids_arr.reshape(n).tolist()

        if self._task == "segment" and masks is not None and n:
            masks_arr = np.asarray(masks)
            if masks_arr.dtype == object:
                masks_arr = np.stack([np.asarray(m) for m in masks_arr.tolist()], axis=0)
            if masks_arr.ndim == 3 and masks_arr.shape[1:] != image.shape[:2]:
                import cv2

                h, w = image.shape[:2]
                resized = np.zeros((masks_arr.shape[0], h, w), dtype=np.uint8)
                for i in range(masks_arr.shape[0]):
                    resized[i] = cv2.resize(
                        masks_arr[i].astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST
                    )
                masks_arr = resized
            image_masks = [_mask_to_polygons(masks_arr[i]) for i in range(n)]
        else:
            image_masks = [[[]] for _ in range(n)]

        return bboxes, detection_scores, ids, image_masks

    def predict(
        self,
        input: DetectionInput,
        score_threshold: float,
        classification_top_n: int = 1,
    ) -> DetectionResult:
        images = self._preprocess_input(input)
        bboxes: List[List[List[int]]] = []
        detection_scores: List[List[float]] = []
        class_ids: List[List[int]] = []
        masks: List[List[List[List[Tuple[int, int]]]]] = []
        keypoints: List[List[List[Tuple[int, int]]]] = []

        for image in images:
            image_boxes, image_scores, image_ids, image_masks = self._predict_one(image, score_threshold)
            bboxes.append(image_boxes)
            detection_scores.append(image_scores)
            class_ids.append(image_ids)
            masks.append(image_masks)
            keypoints.append(_empty_keypoints(len(image_boxes)))

        if self.class_names is not None:
            if classification_top_n > 1:
                raise NotImplementedError("Not implemented for classification_top_n > 1")
            labels_top_n = [
                [[self._label_for_class_id(cid)] for cid in image_ids] for image_ids in class_ids
            ]
            classification_scores_top_n = [[[score] for score in scores] for scores in detection_scores]
        else:
            labels_top_n = [[[str(cid)] for cid in image_ids] for image_ids in class_ids]
            classification_scores_top_n = [[[score] for score in scores] for scores in detection_scores]

        return DetectionResult(
            bboxes=bboxes,
            keypoints=keypoints,
            masks=masks,
            detection_scores=detection_scores,
            labels_top_n=labels_top_n,
            classification_scores_top_n=classification_scores_top_n,
            keypoints_scores=[[None for _ in image_boxes] for image_boxes in bboxes],
            keypoints_labels=None,
        )

    def preprocess_input(self, input: DetectionInput) -> DetectionInput:
        return self._preprocess_input(input)

    @property
    def input_size(self) -> Tuple[int, int]:
        return self._input_size
