import numpy as np

from cv_pipeliner.inferencers.base import Runtime
from cv_pipeliner.inferencers.classification.core import ClassificationRuntime, ClassificationModelSpec
from cv_pipeliner.inferencers.results import ClassificationResult, DetectionResult, EmbeddingResult, KeypointsResult


class FakeClassificationModelSpec(ClassificationModelSpec):
    id: str = None

    @property
    def runtime_cls(self):
        return FakeClassificationRuntime


class FakeClassificationRuntime(ClassificationRuntime):
    def predict(self, input, top_n: int = 1):
        return [["label"] * top_n for _ in input], [[1.0] * top_n for _ in input]

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self):
        return (1, 1)

    @property
    def class_names(self):
        return ["label"]


def test_model_spec_cache_by_id():
    spec = FakeClassificationModelSpec(id="cached-test-model")
    first_runtime = spec.load_runtime()
    second_runtime = spec.load_runtime()

    try:
        assert first_runtime is second_runtime
    finally:
        Runtime._loaded_runtimes = [
            runtime for runtime in Runtime._loaded_runtimes if runtime.spec.id != "cached-test-model"
        ]


def test_classification_result_tuple_roundtrip():
    result = ClassificationResult.from_tuple(([["a", "b"]], [[0.8, 0.2]]))

    assert result.labels_top_n == [["a", "b"]]
    assert result.scores_top_n == [[0.8, 0.2]]
    assert result.as_tuple() == ([["a", "b"]], [[0.8, 0.2]])


def test_detection_result_tuple_roundtrip():
    output = (
        [[(1, 2, 3, 4)]],
        [[[(-1, -1)]]],
        [[[[(0, 0), (1, 1)]]]],
        [[0.9]],
        [[["class"]]],
        [[[0.7]]],
    )
    result = DetectionResult.from_tuple(output)

    assert result.bboxes == [[(1, 2, 3, 4)]]
    assert result.as_tuple() == output


def test_other_result_dataclasses():
    embedding = np.array([1.0, 2.0])

    assert EmbeddingResult([embedding]).embeddings[0] is embedding
    assert KeypointsResult([[(1, 2)]]).keypoints == [[(1, 2)]]
