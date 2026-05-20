import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.inferencers.classification import ClassificationInferencer
from cv_pipeliner.inferencers.classification.core import ClassificationRuntime, ClassificationModelSpec


class FakeClassificationModelSpec(ClassificationModelSpec):
    @property
    def runtime_cls(self):
        return FakeClassificationRuntime


class FakeClassificationRuntime(ClassificationRuntime):
    def predict(self, input, top_n: int = 1):
        labels = [["class-a", "class-b"][:top_n] for _ in input]
        scores = [[0.8, 0.2][:top_n] for _ in input]
        return labels, scores

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self):
        return (4, 4)

    @property
    def class_names(self):
        return ["class-a", "class-b"]


def test_classification_inferencer_predicts_images_data():
    inferencer = ClassificationInferencer(FakeClassificationRuntime(FakeClassificationModelSpec()))

    result = inferencer.predict(
        [ImageData(image=np.zeros((4, 4, 3), dtype=np.uint8))],
        top_n=2,
        open_images_in_data=True,
        disable_tqdm=True,
    )

    assert result[0].label == "class-a"
    assert result[0].classification_score == 0.8
    assert result[0].labels_top_n == ["class-a", "class-b"]
    assert result[0].image is not None


def test_classification_inferencer_predicts_bboxes_and_restores_groups():
    inferencer = ClassificationInferencer(FakeClassificationRuntime(FakeClassificationModelSpec()))
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    grouped_bboxes = [
        [BboxData(image=image, xmin=0, ymin=0, xmax=4, ymax=4)],
        [
            BboxData(image=image, xmin=1, ymin=1, xmax=5, ymax=5),
            BboxData(image=image, xmin=2, ymin=2, xmax=6, ymax=6),
        ],
    ]

    result = inferencer.predict(grouped_bboxes, top_n=1, open_images_in_data=True, disable_tqdm=True)

    assert len(result) == 2
    assert [len(group) for group in result] == [1, 2]
    assert result[0][0].label == "class-a"
    assert result[0][0].cropped_image is not None
