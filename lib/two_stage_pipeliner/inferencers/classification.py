from typing import List

from two_stage_pipeliner.core.data import BboxData

from two_stage_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from two_stage_pipeliner.inference_models.classification.core import ClassificationModel
from two_stage_pipeliner.core.inferencer import Inferencer


class ClassificationInferencer(Inferencer):
    def __init__(self, model: ClassificationModel):
        assert isinstance(model, ClassificationModel)
        super().__init__(model)

    def _postprocess_predictions(
        self,
        n_bboxes_data: List[List[BboxData]],
        n_pred_labels: List[List[str]],
        n_pred_scores: List[List[float]]
    ) -> List[List[BboxData]]:

        n_pred_bboxes_data = []
        for bboxes_data, pred_labels, pred_scores in zip(n_bboxes_data, n_pred_labels, n_pred_scores):
            bboxes_data_res = []
            for (bbox_data, pred_label, pred_classification_score) in zip(bboxes_data, pred_labels, pred_scores):
                bboxes_data_res.append(BboxData(
                    image_path=bbox_data.image_path,
                    image_bytes=bbox_data.image_bytes,
                    cropped_image=bbox_data.cropped_image,
                    xmin=bbox_data.xmin,
                    ymin=bbox_data.ymin,
                    xmax=bbox_data.xmax,
                    ymax=bbox_data.ymax,
                    detection_score=bbox_data.detection_score,
                    label=pred_label,
                    classification_score=pred_classification_score
                ))
            n_pred_bboxes_data.append(bboxes_data_res)

        return n_pred_bboxes_data

    def predict(self, n_bboxes_data_gen: BatchGeneratorBboxData) -> List[List[BboxData]]:
        n_pred_bboxes_data = []
        for n_bboxes_data in n_bboxes_data_gen:
            input = [
                [bbox_data.cropped_image for bbox_data in bboxes_data]
                for bboxes_data in n_bboxes_data
            ]
            input = self.model.preprocess_input(input)
            n_pred_labels, n_pred_scores = self.model.predict(input)
            n_pred_bboxes_data_batch = self._postprocess_predictions(
                n_bboxes_data, n_pred_labels, n_pred_scores
            )
            n_pred_bboxes_data.extend(n_pred_bboxes_data_batch)
        return n_pred_bboxes_data

    @property
    def class_names(self):
        return self.model.class_names
