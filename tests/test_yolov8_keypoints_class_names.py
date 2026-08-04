import numpy as np
import pytest

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inferencers.detection.yolov8 import YOLOv8Runtime, YOLOv8_ModelSpec
from cv_pipeliner.inferencers.postprocess import build_detection_images_data
from cv_pipeliner.inferencers.results import DetectionResult


def test_build_keypoints_labels_from_class_names():
    runtime = YOLOv8Runtime.__new__(YOLOv8Runtime)
    runtime.keypoints_class_names = np.array(["nose", "left_eye", "right_eye"])

    labels = runtime._build_keypoints_labels(
        [
            [
                [(1, 1), (2, 2), (3, 3)],
                [],
            ]
        ]
    )
    assert labels == [[["nose", "left_eye", "right_eye"], None]]


def test_build_keypoints_labels_length_mismatch_raises():
    runtime = YOLOv8Runtime.__new__(YOLOv8Runtime)
    runtime.keypoints_class_names = np.array(["nose", "left_eye"])

    with pytest.raises(ValueError, match="keypoints_class_names length"):
        runtime._build_keypoints_labels([[[(1, 1), (2, 2), (3, 3)]]])


def test_build_detection_images_data_applies_keypoints_labels():
    images_data = [ImageData(image=np.zeros((10, 10, 3), dtype=np.uint8))]
    detection_result = DetectionResult(
        bboxes=[[(1, 2, 5, 6)]],
        keypoints=[[[ (2, 3), (4, 5) ]]],
        masks=[[[[]]]],
        detection_scores=[[0.9]],
        labels_top_n=[[["person"]]],
        classification_scores_top_n=[[[0.9]]],
        keypoints_scores=[[[0.8, 0.7]]],
        keypoints_labels=[[["nose", "ear"]]],
    )
    pred = build_detection_images_data(
        images_data=images_data,
        detection_result=detection_result,
        open_images_in_images_data=False,
        open_cropped_images_in_bboxes_data=False,
    )
    assert pred[0].bboxes_data[0].keypoints_labels == ["nose", "ear"]
    assert pred[0].bboxes_data[0].label == "person"


def test_build_keypoints_labels_none_when_class_names_missing():
    runtime = YOLOv8Runtime.__new__(YOLOv8Runtime)
    runtime.keypoints_class_names = None

    assert runtime._build_keypoints_labels([[[(1, 1), (2, 2)]]]) is None


def test_yolov8_model_spec_keypoints_class_names_default_none():
    spec = YOLOv8_ModelSpec(model_path="yolov8n-pose.pt", class_names=["person"])
    assert spec.keypoints_class_names is None


def test_yolov8_model_spec_accepts_keypoints_class_names():
    spec = YOLOv8_ModelSpec(
        model_path="yolov8n-pose.pt",
        class_names=["person"],
        keypoints_class_names=["nose", "left_eye", "right_eye"],
    )
    assert spec.keypoints_class_names == ["nose", "left_eye", "right_eye"]
