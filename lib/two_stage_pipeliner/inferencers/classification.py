from typing import List

from two_stage_pipeliner.core.data import BboxData

from two_stage_pipeliner.core.batch_generator import BatchGeneratorBboxData
from two_stage_pipeliner.inference_models.classification.core import ClassificationModel
from two_stage_pipeliner.core.inferencer import Inferencer


class ClassificationInferencer(Inferencer):
    def __init__(self, model: ClassificationModel):
        assert isinstance(model, ClassificationModel)
        super(ClassificationInferencer, self).__init__(model)

    def predict(self, data_generator: BatchGeneratorBboxData) -> List[List[BboxData]]:
        bboxes_data = []
        n_bboxes_data = []
        for batch in data_generator:
            input = [bbox_data.image_bbox for bbox_data in batch]
            input = self.model.preprocess_input(input)
            n_results = self.model.predict(input)
            for bbox_data, results in zip(batch, n_results):
                bboxes_data = []
                for (
                    pred_label,
                    pred_classification_score,
                ) in results:
                    bboxes_data.append(BboxData(
                        image_path=bbox_data.image_path,
                        image_bbox=bbox_data.image_bbox,
                        xmin=bbox_data.xmin,
                        ymin=bbox_data.ymin,
                        xmax=bbox_data.xmax,
                        ymax=bbox_data.ymax,
                        detection_score=bbox_data.detection_score,
                        label=pred_label,
                        classification_score=pred_classification_score
                    ))
                n_bboxes_data.append(bboxes_data)
        return n_bboxes_data
