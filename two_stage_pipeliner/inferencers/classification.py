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
        bboxes_data: List[BboxData],
        pred_labels: List[str],
        pred_scores: List[float],
        open_cropped_images_in_bboxes_data: bool
    ) -> List[List[BboxData]]:
        bboxes_data_res = []
        for (bbox_data, pred_label, pred_classification_score) in zip(bboxes_data, pred_labels, pred_scores):
            cropped_image = bbox_data.cropped_image if open_cropped_images_in_bboxes_data else None
            bboxes_data_res.append(BboxData(
                image_path=bbox_data.image_path,
                image_bytes=bbox_data.image_bytes,
                cropped_image=cropped_image,
                xmin=bbox_data.xmin,
                ymin=bbox_data.ymin,
                xmax=bbox_data.xmax,
                ymax=bbox_data.ymax,
                detection_score=bbox_data.detection_score,
                label=pred_label,
                classification_score=pred_classification_score
            ))

        return bboxes_data_res

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
        bboxes_data_gen: BatchGeneratorBboxData,
        open_cropped_images_in_bboxes_data: bool = False
    ) -> List[List[BboxData]]:
        pred_bboxes_data = []
        for bboxes_data in bboxes_data_gen:
            input = [bbox_data.cropped_image for bbox_data in bboxes_data]
            input = self.model.preprocess_input(input)
            pred_labels, pred_scores = self.model.predict(input)
            pred_bboxes_data.extend(self._postprocess_predictions(
                bboxes_data=bboxes_data,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                open_cropped_images_in_bboxes_data=open_cropped_images_in_bboxes_data
            ))
        n_pred_bboxes_data = self._split_chunks(pred_bboxes_data, bboxes_data_gen.shapes)
        return n_pred_bboxes_data

    @property
    def class_names(self):
        return self.model.class_names
