import numpy as np

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inferencers.classification.core import ClassificationRuntime, ClassificationModelSpec
from cv_pipeliner.inferencers.detection.core import DetectionRuntime, DetectionModelSpec
from cv_pipeliner.inferencers.pipeline import PipelineModelSpec


class FakeDetectionModelSpec(DetectionModelSpec):
    class_names: list = ["detector-class"]

    @property
    def runtime_cls(self):
        return FakeDetectionRuntime


class FakeDetectionRuntime(DetectionRuntime):
    def __init__(self, model_spec: FakeDetectionModelSpec):
        super().__init__(model_spec)

    def predict(self, input, score_threshold: float, classification_top_n: int = None):
        return (
            [[(1, 1, 5, 5)] for _ in input],
            [[[(-1, -1)]] for _ in input],
            [[[[(1, 1), (5, 1), (5, 5)]]] for _ in input],
            [[0.7] for _ in input],
            [[["detector-class"]] for _ in input],
            [[[0.7]] for _ in input],
        )

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self):
        return (8, 8)


class FakeClassificationModelSpec(ClassificationModelSpec):
    @property
    def runtime_cls(self):
        return FakeClassificationRuntime


class FakeClassificationRuntime(ClassificationRuntime):
    def predict(self, input, top_n: int = 1):
        return [["classifier-class"] * top_n for _ in input], [[0.95] * top_n for _ in input]

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self):
        return (4, 4)

    @property
    def class_names(self):
        return ["classifier-class"]


def test_pipeline_inferencer_detection_only_uses_detector_labels():
    inferencer = PipelineModelSpec(detection_model_spec=FakeDetectionModelSpec()).load_pipeline_inferencer()

    result = inferencer.predict(
        [ImageData(image=np.zeros((8, 8, 3), dtype=np.uint8))],
        detection_score_threshold=0.1,
        disable_tqdm=True,
        disable_tqdm_classification=True,
    )

    assert result[0].bboxes_data[0].label == "detector-class"
    assert result[0].bboxes_data[0].classification_score == 0.7


def test_pipeline_inferencer_classifies_detected_bboxes():
    inferencer = PipelineModelSpec(
        detection_model_spec=FakeDetectionModelSpec(),
        classification_model_spec=FakeClassificationModelSpec(),
    ).load_pipeline_inferencer()

    result = inferencer.predict(
        [ImageData(image=np.zeros((8, 8, 3), dtype=np.uint8))],
        detection_score_threshold=0.1,
        classification_top_n=1,
        disable_tqdm=True,
        disable_tqdm_classification=True,
    )

    assert result[0].bboxes_data[0].label == "classifier-class"
    assert result[0].bboxes_data[0].classification_score == 0.95
    assert inferencer.detection_inferencer is not None
    assert inferencer.classification_inferencer is not None
