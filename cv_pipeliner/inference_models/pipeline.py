from typing import List, Tuple, Type
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from cv_pipeliner.core.inference_model import InferenceModel, ModelSpec
from cv_pipeliner.inference_models.detection.core import DetectionModelSpec, DetectionModel
from cv_pipeliner.inference_models.classification.core import ClassificationModelSpec, ClassificationModel

from cv_pipeliner.utils.images import cut_bboxes_from_image

from cv_pipeliner.logging import logger


@dataclass
class PipelineModelSpec(ModelSpec):
    detection_model_spec: DetectionModelSpec
    classification_model_spec: ClassificationModelSpec

    @property
    def inference_model_cls(self) -> Type['PipelineModel']:
        from cv_pipeliner.inference_models.pipeline import PipelineModel
        return PipelineModel


Bbox = Tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)
Score = float
Label = str

Bboxes = List[Bbox]
DetectionScores = List[Score]
Labels = List[Label]
ClassificationScores = List[Score]

PipelineInput = List[np.ndarray]
PipelineOutput = List[
    Tuple[
        List[Bboxes],
        List[DetectionScores],
        List[Labels],
        List[ClassificationScores]
    ]
]


class PipelineModel(InferenceModel):
    def __init__(self, model_spec: PipelineModelSpec = None):
        if model_spec is not None:
            isinstance(model_spec, PipelineModelSpec)
            super().__init__(model_spec)
            self.detection_model = model_spec.detection_model_spec.load()
            self.classification_model = model_spec.classification_model_spec.load()

    def load_from_loaded_models(
        self,
        detection_model: DetectionModel,
        classification_model: ClassificationModel
    ):
        isinstance(detection_model, DetectionModel)
        isinstance(classification_model, ClassificationModel)
        self._model_spec = PipelineModelSpec(
            detection_model_spec=detection_model.model_spec,
            classification_model_spec=classification_model.model_spec
        )
        self.detection_model = detection_model
        self.classification_model = classification_model

    def _split_chunks(self,
                      _list: List,
                      shapes: List[int]) -> List:
        cnt = 0
        chunks = []
        for shape in shapes:
            chunks.append(_list[cnt: cnt + shape])
            cnt += shape
        return chunks

    def predict(
        self,
        input: PipelineInput,
        detection_score_threshold: float,
        classification_top_n: int = 1,
        classification_batch_size: int = 16
    ) -> PipelineOutput:
        logger.info("Running detection...")
        detection_input = self.detection_model.preprocess_input(input)
        _, n_pred_bboxes, n_pred_detection_scores = self.detection_model.predict(
            detection_input,
            score_threshold=detection_score_threshold
        )
        logger.info(
            f"Detection: found {np.sum([len(pred_bboxes) for pred_bboxes in n_pred_bboxes])} bboxes!"
        )

        logger.info("Running classification...")
        shapes = [len(pred_bboxes) for pred_bboxes in n_pred_bboxes]
        pred_labels_top_n, pred_classification_scores_top_n = [], []
        with tqdm(total=np.sum(shapes)) as pbar:
            for image, pred_bboxes in zip(input, n_pred_bboxes):
                pred_bboxes_batches = np.array_split(pred_bboxes, max(1, len(pred_bboxes) // classification_batch_size))
                for pred_bboxes_batch in pred_bboxes_batches:
                    pred_cropped_images_batch = cut_bboxes_from_image(image, pred_bboxes)
                    classification_input = self.classification_model.preprocess_input([
                        cropped_image
                        for cropped_image in pred_cropped_images_batch
                    ])
                    pred_labels_top_n_batch, pred_classification_scores_top_n_batch = self.classification_model.predict(
                        input=classification_input,
                        top_n=classification_top_n
                    )
                    pred_labels_top_n.extend(pred_labels_top_n_batch)
                    pred_classification_scores_top_n.extend(pred_classification_scores_top_n_batch)
                    pbar.update(len(pred_bboxes_batch))
        n_pred_labels_top_n = self._split_chunks(pred_labels_top_n, shapes)
        n_pred_classification_scores_top_n = self._split_chunks(pred_classification_scores_top_n, shapes)
        logger.info("Classification end!")
        return (
            n_pred_bboxes,
            n_pred_detection_scores,
            n_pred_labels_top_n,
            n_pred_classification_scores_top_n
        )

    def preprocess_input(self, input):
        return input

    @property
    def input_size(self) -> Tuple[int, int]:
        return self.detection_model.input_size

    @property
    def class_names(self) -> List[str]:
        return self.classification_model.class_names
