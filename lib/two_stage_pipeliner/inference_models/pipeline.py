from typing import List, Tuple
import numpy as np

from two_stage_pipeliner.core.inference_model import InferenceModel
from two_stage_pipeliner.core.inference_models.detection import DetectionModel
from two_stage_pipeliner.core.inference_models.classification import ClassificationModel

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


class PipelineModel(InferenceModel):
    def __init__(self,
                 detection_model: DetectionModel,
                 classification_model: ClassificationModel):
        self.detection_model = detection_model
        self.classification_model = classification_model
        super(PipelineModel, self).__init__()

    def predict(self, input: PipelineInput) -> PipelineOutput:
        (n_pred_img_bboxes, n_pred_bboxes,
         n_pred_detection_scores) = self.detection_model.predict(input)
        n_pred_labels, n_pred_classification_scores = self.classification_model.predict(
            n_pred_img_bboxes
        )
        return (
            n_pred_img_bboxes,
            n_pred_bboxes,
            n_pred_detection_scores,
            n_pred_labels,
            n_pred_classification_scores
        )
