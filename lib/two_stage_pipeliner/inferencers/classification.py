from typing import List

from two_stage_pipeliner.core.data import BboxData

from two_stage_pipeliner.core.batch_generator import BatchGeneratorBboxData
from two_stage_pipeliner.inference_models.classification.core import ClassificationModel
from two_stage_pipeliner.core.inferencer import Inferencer


class ClassificationInferencer(Inferencer):
    def __init__(self, model: ClassificationModel):
        assert isinstance(model, ClassificationModel)
        Inferencer.__init__(self, model)

    def predict(self, bboxes_data_gen: BatchGeneratorBboxData) -> List[List[BboxData]]:
        n_pred_bboxes_data = []
        for batch in bboxes_data_gen:
            input = [
                self.model.preprocess_input([bbox_data.cropped_image for bbox_data in bboxes_data])
                for bboxes_data in batch
            ]
            n_pred_labels, n_pred_scores = self.model.predict(input)
            for bboxes_data, pred_labels, pred_scores in zip(
                batch, n_pred_labels, n_pred_scores
            ):
                bboxes_data_res = []
                for (
                    bbox_data,
                    pred_label,
                    pred_classification_score,
                ) in zip(bboxes_data, pred_labels, pred_scores):
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
