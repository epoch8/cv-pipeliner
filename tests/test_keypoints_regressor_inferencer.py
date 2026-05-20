import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.inferencers.keypoints_regressor import KeypointsRegressorInferencer
from cv_pipeliner.inferencers.keypoints_regressor.core import (
    KeypointsRegressorRuntime,
    KeypointsRegressorModelSpec,
)


class FakeKeypointsRegressorModelSpec(KeypointsRegressorModelSpec):
    @property
    def runtime_cls(self):
        return FakeKeypointsRegressorRuntime


class FakeKeypointsRegressorRuntime(KeypointsRegressorRuntime):
    def predict(self, input):
        return [[(1, 2), (image.shape[1] - 1, image.shape[0] - 1)] for image in input]

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self):
        return (4, 4)


def test_keypoints_regressor_predicts_images_data_and_progress():
    inferencer = KeypointsRegressorInferencer(FakeKeypointsRegressorRuntime(FakeKeypointsRegressorModelSpec()))
    progress = []
    images_data = [
        ImageData(image=np.zeros((4, 5, 3), dtype=np.uint8)),
        ImageData(image=np.zeros((6, 7, 3), dtype=np.uint8)),
    ]

    result = inferencer.predict(
        images_data,
        open_images_in_data=True,
        batch_size_default=1,
        disable_tqdm=True,
        progress_callback=progress.append,
    )

    assert progress == [1, 2]
    assert result[0].image is not None
    np.testing.assert_array_equal(result[0].keypoints, np.array([[1, 2], [4, 3]]))
    np.testing.assert_array_equal(result[1].keypoints, np.array([[1, 2], [6, 5]]))


def test_keypoints_regressor_predicts_bboxes_and_restores_groups():
    inferencer = KeypointsRegressorInferencer(FakeKeypointsRegressorRuntime(FakeKeypointsRegressorModelSpec()))
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    grouped_bboxes = [
        [BboxData(image=image, xmin=0, ymin=0, xmax=4, ymax=4)],
        [
            BboxData(image=image, xmin=1, ymin=1, xmax=5, ymax=6),
            BboxData(image=image, xmin=2, ymin=2, xmax=8, ymax=9),
        ],
    ]

    result = inferencer.predict(grouped_bboxes, open_images_in_data=True, batch_size_default=2, disable_tqdm=True)

    assert [len(group) for group in result] == [1, 2]
    assert result[0][0].cropped_image is not None
    np.testing.assert_array_equal(result[0][0].keypoints, np.array([[1, 2], [3, 3]]))
    np.testing.assert_array_equal(result[1][1].keypoints, np.array([[1, 2], [5, 6]]))
