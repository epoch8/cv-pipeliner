"""Unit tests for RF-DETR ModelSpec / Runtime without requiring rfdetr weights."""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from cv_pipeliner.inferencers.detection.rfdetr import RFDETRRuntime, RFDETR_ModelSpec, _mask_to_polygons


def test_rfdetr_model_spec_defaults_and_runtime_cls():
    spec = RFDETR_ModelSpec(
        model_path="/tmp/fake.pth",
        class_names=["a", "b"],
        input_size=(640, 640),
        variant="medium",
        task="detect",
    )
    assert spec.runtime_cls is RFDETRRuntime
    assert spec.task == "detect"
    assert spec.variant == "medium"


def test_rfdetr_package_export():
    from cv_pipeliner import RFDETR_ModelSpec as Exported

    assert Exported is RFDETR_ModelSpec


def test_mask_to_polygons_empty_and_box():
    empty = np.zeros((8, 8), dtype=np.uint8)
    assert _mask_to_polygons(empty) == [[]]

    filled = np.zeros((8, 8), dtype=np.uint8)
    filled[2:6, 2:6] = 1
    polys = _mask_to_polygons(filled)
    assert len(polys) >= 1
    assert len(polys[0]) >= 3


class _FakeDetections:
    def __init__(self, n: int = 1, with_mask: bool = False, h: int = 10, w: int = 10):
        self.xyxy = np.array([[1.0, 2.0, 5.0, 6.0]] * n, dtype=float)
        self.confidence = np.array([0.9] * n, dtype=float)
        self.class_id = np.array([0] * n, dtype=int)
        self.mask = None
        if with_mask:
            masks = np.zeros((n, h, w), dtype=np.uint8)
            masks[:, 2:6, 2:6] = 1
            self.mask = masks


class _FakeModel:
    class_names = {0: "damage"}

    def __init__(self, with_mask: bool = False):
        self.with_mask = with_mask

    def predict(self, image, threshold: float = 0.0):
        h, w = image.shape[:2]
        return _FakeDetections(n=1, with_mask=self.with_mask, h=h, w=w)


def _runtime_without_load(task: str, with_mask: bool = False) -> RFDETRRuntime:
    runtime = RFDETRRuntime.__new__(RFDETRRuntime)
    runtime._spec = RFDETR_ModelSpec(
        model_path="/tmp/fake.pth",
        class_names=["damage"],
        input_size=32,
        task=task,  # type: ignore[arg-type]
        variant="medium",
    )
    runtime.class_names = np.array(["damage"])
    runtime._task = task
    runtime._variant = "medium"
    runtime._input_size = (32, 32)
    runtime._preprocess_input = lambda images: list(images)
    runtime.model = _FakeModel(with_mask=with_mask)
    return runtime


def test_rfdetr_runtime_predict_detect_without_weights():
    runtime = _runtime_without_load("detect", with_mask=False)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = runtime.predict([image], score_threshold=0.1)
    assert len(result.bboxes) == 1
    assert result.bboxes[0] == [[1, 2, 5, 6]]
    assert result.detection_scores[0] == [pytest.approx(0.9)]
    assert result.labels_top_n[0] == [["damage"]]
    assert result.masks[0] == [[[]]]


def test_rfdetr_runtime_predict_segment_fills_polygons():
    runtime = _runtime_without_load("segment", with_mask=True)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    result = runtime.predict([image], score_threshold=0.1)
    assert len(result.masks[0]) == 1
    assert len(result.masks[0][0]) >= 1
    assert len(result.masks[0][0][0]) >= 3


def test_rfdetr_load_requires_model_path():
    runtime = RFDETRRuntime.__new__(RFDETRRuntime)
    runtime._task = "detect"
    runtime._variant = "medium"
    with pytest.raises(ValueError, match="model_path"):
        runtime._load_model(RFDETR_ModelSpec(model_path=None, task="detect", variant="medium"))


def test_rfdetr_import_compatibility():
    from cv_pipeliner.inferencers.detection.rfdetr import RFDETRRuntime, RFDETR_ModelSpec

    assert RFDETRRuntime is not None
    assert RFDETR_ModelSpec is not None

