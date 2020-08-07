from typing import List, Tuple
import numpy as np

from two_stage_pipeliner.core.inference_model import InferenceModel
from two_stage_pipeliner.inference_models.detection.core import DetectionModel
from two_stage_pipeliner.inference_models.classification.core import ClassificationModel

ImageInput = np.ndarray
Bbox = Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
Score = float
Label = str

ImgBboxes = List[ImageInput]
Bboxes = List[Bbox]
DetectionScores = List[Score]
Labels = List[Label]
ClassificationScores = List[Score]

PipelineInput = List[ImageInput]
PipelineOutput = List[
    Tuple[
        List[ImgBboxes],
        List[Bboxes],
        List[DetectionScores],
        List[Labels],
        List[ClassificationScores]
    ]
]


class Pipeline(InferenceModel):
    def __init__(self,
                 detection_model: DetectionModel,
                 classification_model: ClassificationModel):
        super(Pipeline, self).__init__()
        self.load((detection_model, classification_model))

    def load(self, checkpoint):
        detection_model, classification_model = checkpoint
        self.detection_model = detection_model
        self.classification_model = classification_model

    def predict(self, input: PipelineInput) -> PipelineOutput:
        detection_input = self.detection_model.preprocess_input(input)
        (n_pred_img_bboxes, n_pred_bboxes,
         n_pred_detection_scores) = self.detection_model.predict(detection_input)
        classification_input = [
            self.classification_model.preprocess_input([image_bbox for image_bbox in images_bbox])
            for images_bbox in n_pred_img_bboxes
        ]
        n_pred_labels, n_pred_classification_scores = self.classification_model.predict(
            classification_input
        )
        return (
            n_pred_img_bboxes,
            n_pred_bboxes,
            n_pred_detection_scores,
            n_pred_labels,
            n_pred_classification_scores
        )

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self) -> int:
        return self.model_spec.input_size
