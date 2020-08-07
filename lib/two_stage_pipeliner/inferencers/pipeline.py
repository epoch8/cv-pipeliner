from typing import List

from two_stage_pipeliner.core.data import BboxData, ImageData
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData

from two_stage_pipeliner.core.inferencer import Inferencer
from two_stage_pipeliner.inference_models.pipeline import PipelineModel


class PipelineInferencer(Inferencer):
    def __init__(self, model: PipelineModel):
        assert isinstance(model, PipelineModel)
        super(PipelineInferencer, self).__init__(model)

    def predict(self, data_generator: BatchGeneratorImageData) -> List[ImageData]:
        images_data = []
        for batch in data_generator:
            input = [image_data.image for image_data in batch]
            n_results = self.model.predict(input)
            for image_data, results in zip(batch, n_results):
                bboxes_data = []
                for (
                    pred_image_bbox,
                    pred_bbox,
                    pred_detection_score,
                    pred_label,
                    pred_classification_score
                ) in results:
                    ymin, xmin, ymax, xmax = pred_bbox
                    bboxes_data.append(BboxData(
                        image_path=image_data.image_path,
                        image_bbox=pred_image_bbox,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        detection_score=pred_detection_score,
                        label=pred_label,
                        classification_score=pred_classification_score
                    ))
            images_data.append(ImageData(
                image_path=image_data.image_path,
                bboxes_data=bboxes_data
            ))

        return images_data
