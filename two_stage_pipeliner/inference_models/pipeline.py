from typing import List, Tuple, ClassVar
from dataclasses import dataclass
import numpy as np

from two_stage_pipeliner.core.inference_model import InferenceModel, ModelSpec
from two_stage_pipeliner.inference_models.detection.core import DetectionModelSpec
from two_stage_pipeliner.inference_models.classification.core import ClassificationModelSpec


@dataclass
class PipelineModelSpec(ModelSpec):
    detection_model_spec: DetectionModelSpec
    classification_model_spec: ClassificationModelSpec

    @property
    def inference_model(self) -> ClassVar['PipelineModel']:
        from two_stage_pipeliner.inference_models.pipeline import PipelineModel
        return PipelineModel


Bbox = Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
Score = float
Label = str

CroppedImages = List[np.ndarray]
Bboxes = List[Bbox]
DetectionScores = List[Score]
Labels = List[Label]
ClassificationScores = List[Score]

PipelineInput = List[np.ndarray]
PipelineOutput = List[
    Tuple[
        List[CroppedImages],
        List[Bboxes],
        List[DetectionScores],
        List[Labels],
        List[ClassificationScores]
    ]
]


class PipelineModel(InferenceModel):
    def load(self, model_spec: PipelineModelSpec):
        isinstance(model_spec, PipelineModelSpec)
        super().load(model_spec)
        self.detection_model = model_spec.detection_model_spec.load()
        self.classification_model = model_spec.classification_model_spec.load()

    def predict(self, input: PipelineInput,
                detection_score_threshold: float) -> PipelineOutput:
        detection_input = self.detection_model.preprocess_input(input)
        (n_pred_cropped_images, n_pred_bboxes,
         n_pred_detection_scores) = self.detection_model.predict(
            detection_input,
            score_threshold=detection_score_threshold
        )
        classification_input = self.classification_model.preprocess_input([
            [cropped_image for cropped_image in pred_cropped_images]
            for pred_cropped_images in n_pred_cropped_images
        ])
        n_pred_labels, n_pred_classification_scores = self.classification_model.predict(
            classification_input
        )
        return (
            n_pred_cropped_images,
            n_pred_bboxes,
            n_pred_detection_scores,
            n_pred_labels,
            n_pred_classification_scores
        )

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.detection_model.input_size

    @property
    def class_names(self) -> List[str]:
        return self.classification_model.class_names
