import numpy as np

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inferencers.detection import DetectionInferencer
from cv_pipeliner.inferencers.detection.core import DetectionRuntime, DetectionModelSpec


class FakeDetectionModelSpec(DetectionModelSpec):
    class_names: list = ["detected"]

    @property
    def runtime_cls(self):
        return FakeDetectionRuntime


class FakeDetectionRuntime(DetectionRuntime):
    def __init__(self, model_spec: FakeDetectionModelSpec):
        super().__init__(model_spec)

    def predict(self, input, score_threshold: float, classification_top_n: int = None):
        bboxes, keypoints, masks, detection_scores = [], [], [], []
        labels_top_n, classification_scores_top_n = [], []
        for _ in input:
            if score_threshold > 0.95:
                bboxes.append([])
                keypoints.append([])
                masks.append([])
                detection_scores.append([])
                labels_top_n.append([])
                classification_scores_top_n.append([])
            else:
                bboxes.append([(1, 2, 5, 6)])
                keypoints.append([[(1, 2)]])
                masks.append([[[(1, 2), (5, 2), (5, 6)]]])
                detection_scores.append([0.9])
                labels_top_n.append([["detected"]])
                classification_scores_top_n.append([[0.9]])
        return bboxes, keypoints, masks, detection_scores, labels_top_n, classification_scores_top_n

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self):
        return (10, 10)


def test_detection_inferencer_reconstructs_image_data_and_progress():
    inferencer = DetectionInferencer(FakeDetectionRuntime(FakeDetectionModelSpec()))
    progress = []
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    result = inferencer.predict(
        [ImageData(image=image)],
        score_threshold=0.5,
        open_images_in_images_data=True,
        open_cropped_images_in_bboxes_data=True,
        disable_tqdm=True,
        progress_callback=progress.append,
    )

    assert progress == [1]
    assert result[0].image is not None
    assert result[0].bboxes_data[0].coords == (1, 2, 5, 6)
    assert result[0].bboxes_data[0].detection_score == 0.9
    assert result[0].bboxes_data[0].cropped_image is not None


def test_detection_inferencer_handles_empty_detections():
    inferencer = DetectionInferencer(FakeDetectionRuntime(FakeDetectionModelSpec()))

    result = inferencer.predict([ImageData(image=np.zeros((10, 10, 3), dtype=np.uint8))], 0.99, disable_tqdm=True)

    assert result[0].bboxes_data == []
    assert result[0].image is None
