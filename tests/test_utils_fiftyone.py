import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.utils import fiftyone as fiftyone_utils
from cv_pipeliner.utils.fiftyone import FiftyOneSession, FifyOneSession


def _mock_fiftyone_import(monkeypatch):
    monkeypatch.setattr(FiftyOneSession, "_active_sessions", 0)
    monkeypatch.setattr(FiftyOneSession, "_active_config", None)
    monkeypatch.setattr(FiftyOneSession, "_active_previous_env_values", {})
    monkeypatch.setattr(fiftyone_utils.importlib, "import_module", lambda name: object())


def _mock_fiftyone_module(monkeypatch):
    fo = MagicMock()
    fo.Keypoint = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    fo.Keypoints = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(FiftyOneSession, "_active_sessions", 0)
    monkeypatch.setattr(FiftyOneSession, "_active_config", None)
    monkeypatch.setattr(FiftyOneSession, "_active_previous_env_values", {})
    monkeypatch.setattr(fiftyone_utils.importlib, "import_module", lambda name: fo)
    return fo


def test_fify_one_session_alias_is_preserved():
    assert FifyOneSession is FiftyOneSession


def test_fiftyone_session_restores_previous_environment(monkeypatch):
    _mock_fiftyone_import(monkeypatch)
    monkeypatch.setenv("FIFTYONE_DATABASE_DIR", "previous-dir")
    monkeypatch.delenv("FIFTYONE_DATABASE_URI", raising=False)

    with FiftyOneSession(database_dir="new-dir", database_uri="mongodb://localhost:27017"):
        assert os.environ["FIFTYONE_DATABASE_DIR"] == "new-dir"
        assert os.environ["FIFTYONE_DATABASE_URI"] == "mongodb://localhost:27017"

    assert os.environ["FIFTYONE_DATABASE_DIR"] == "previous-dir"
    assert "FIFTYONE_DATABASE_URI" not in os.environ


def test_fiftyone_session_close_is_idempotent(monkeypatch):
    _mock_fiftyone_import(monkeypatch)

    session = FiftyOneSession(database_name="test")
    session.close()
    session.close()

    assert FiftyOneSession._active_sessions == 0
    assert "FIFTYONE_DATABASE_NAME" not in os.environ


def test_fiftyone_session_allows_same_database_config(monkeypatch):
    _mock_fiftyone_import(monkeypatch)
    session = FiftyOneSession()
    other_session = FiftyOneSession()

    try:
        assert FiftyOneSession._active_sessions == 2
    finally:
        other_session.close()
        session.close()


def test_fiftyone_session_rejects_different_active_database_config(monkeypatch):
    _mock_fiftyone_import(monkeypatch)
    session = FiftyOneSession(database_name="one")

    try:
        with pytest.raises(RuntimeError):
            FiftyOneSession(database_name="two")
    finally:
        session.close()


def test_convert_bbox_data_keypoints_includes_confidences(monkeypatch):
    _mock_fiftyone_module(monkeypatch)
    session = FiftyOneSession()
    try:
        bbox_data = BboxData(
            xmin=0,
            ymin=0,
            xmax=100,
            ymax=100,
            meta_width=200,
            meta_height=200,
            label="person",
            keypoints=np.array([[10, 20], [30, 40]]),
            keypoints_scores=[0.9, 0.8],
        )
        fo_keypoint = session.convert_bbox_data_keypoints_to_fo_keypoint(bbox_data)
        assert fo_keypoint.confidences == [0.9, 0.8]
        assert fo_keypoint.label == "person"
        assert fo_keypoint.source_coords == (0, 0, 100, 100)
    finally:
        session.close()


def test_convert_image_data_keypoints_includes_image_confidences(monkeypatch):
    _mock_fiftyone_module(monkeypatch)
    session = FiftyOneSession()
    try:
        image_data = ImageData(
            image_path="img.jpg",
            meta_width=100,
            meta_height=100,
            label="scene",
            keypoints=np.array([[10, 20], [30, 40]]),
            keypoints_scores=[0.5, 0.6],
        )
        fo_keypoints = session.convert_image_data_to_fo_keypoints(image_data)
        assert len(fo_keypoints.keypoints) == 1
        assert fo_keypoints.keypoints[0].confidences == [0.5, 0.6]
    finally:
        session.close()


def test_convert_sample_to_image_data_restores_keypoints_scores(monkeypatch):
    _mock_fiftyone_module(monkeypatch)
    session = FiftyOneSession()
    try:

        class _ContainsKeypoint(SimpleNamespace):
            def __contains__(self, key):
                return key in {"source_coords", "confidences"}

        fo_detection = SimpleNamespace(bounding_box=[0.0, 0.0, 0.5, 0.5], label="person")
        fo_detection.__getitem__ = lambda key: None
        fo_keypoint = _ContainsKeypoint(
            points=[(0.1, 0.2), (0.3, 0.4)],
            confidences=[0.91, 0.82],
            source_coords=(0, 0, 50, 50),
        )
        fields = {
            "detections": SimpleNamespace(detections=[fo_detection]),
            "keypoints": SimpleNamespace(keypoints=[fo_keypoint]),
        }

        class Sample:
            filepath = "img.jpg"
            metadata = SimpleNamespace(width=100, height=100)

            def has_field(self, name):
                return name in fields

            def __getitem__(self, key):
                return fields[key]

        restored = session.convert_sample_to_image_data(
            Sample(),
            fo_detections_label="detections",
            fo_keypoints_label="keypoints",
        )
        assert len(restored.bboxes_data) == 1
        assert restored.bboxes_data[0].keypoints_scores == pytest.approx([0.91, 0.82])
        np.testing.assert_allclose(restored.bboxes_data[0].keypoints, [[10.0, 20.0], [30.0, 40.0]])
    finally:
        session.close()
