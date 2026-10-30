"""YOLOv5 per-image postprocess with class_names, without weights or network: the real
`YOLOv5Runtime._postprocess_prediction` is fed raw outputs and its result goes through the real
`build_detection_images_data`, so `BboxData` validation is exercised for every bbox."""
import numpy as np
import pytest

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inferencers.detection.yolov5 import YOLOv5Runtime, YOLOv5_ModelSpec
from cv_pipeliner.inferencers.postprocess import build_detection_images_data
from cv_pipeliner.inferencers.results import DetectionResult

CLASS_NAMES = ["a", "b"]


def _runtime_with_class_names() -> YOLOv5Runtime:
    # YOLOv5Runtime.__init__ would load the torch model; postprocess only needs what the base Runtime
    # init stores (the spec) and the class names YOLOv5Runtime.__init__ derives from it
    runtime = YOLOv5Runtime.__new__(YOLOv5Runtime)
    super(YOLOv5Runtime, runtime).__init__(YOLOv5_ModelSpec(model_path="unused.pt", class_names=CLASS_NAMES))
    runtime.class_names = np.array(CLASS_NAMES)
    return runtime


def _postprocess(runtime: YOLOv5Runtime, classification_top_n: int):
    return runtime._postprocess_prediction(
        raw_bboxes=np.array([[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]]),
        raw_keypoints=np.zeros((2, 0, 2)),
        raw_scores=np.array([0.8, 0.6]),
        raw_classes=np.array([1, 0]),
        score_threshold=0.5,
        current_width=100,
        current_height=100,
        target_width=100,
        target_height=100,
        classification_top_n=classification_top_n,
    )


def test_class_scores_are_per_bbox_top_n_and_become_float_bbox_scores():
    result = _postprocess(_runtime_with_class_names(), classification_top_n=1)

    assert np.asarray(result.classification_scores_top_n).shape == (2, 1)
    images_data = build_detection_images_data(
        images_data=[ImageData(image=np.zeros((100, 100, 3), dtype=np.uint8))],
        detection_result=DetectionResult(
            bboxes=[result.bboxes],
            keypoints=[result.keypoints],
            masks=[[[] for _ in result.bboxes]],
            detection_scores=[result.detection_scores],
            labels_top_n=[result.labels_top_n],
            classification_scores_top_n=[result.classification_scores_top_n],
        ),
        open_images_in_images_data=False,
        open_cropped_images_in_bboxes_data=False,
    )

    bboxes_data = images_data[0].bboxes_data
    assert [bbox_data.labels_top_n for bbox_data in bboxes_data] == [["b"], ["a"]]
    for bbox_data in bboxes_data:
        assert isinstance(bbox_data.classification_score, float)
        assert bbox_data.classification_score == pytest.approx(bbox_data.detection_score)
        assert bbox_data.classification_scores_top_n == pytest.approx([bbox_data.detection_score])


def test_class_names_with_top_n_above_one_is_not_implemented():
    with pytest.raises(NotImplementedError):
        _postprocess(_runtime_with_class_names(), classification_top_n=2)
