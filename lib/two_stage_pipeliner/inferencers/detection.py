from typing import List

from two_stage_pipeliner.core.data import BboxData, ImageData
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.inference_models.detection.core import DetectionModel
from two_stage_pipeliner.core.inferencer import Inferencer


class DetectionInferencer(Inferencer):
    def __init__(self, model: DetectionModel):
        assert isinstance(model, DetectionModel)
        Inferencer.__init__(self, model)

    def predict(self, data_generator: BatchGeneratorImageData,
                score_threshold: float) -> List[ImageData]:
        images_data = []
        for batch in data_generator:
            input = [image_data.image for image_data in batch]
            input = self.model.preprocess_input(input)
            n_img_boxes, n_pred_bboxes, n_pred_scores = self.model.predict(
                input,
                score_threshold=score_threshold
            )
            for image_data, img_boxes, pred_bboxes, pred_scores in zip(
                batch, n_img_boxes, n_pred_bboxes, n_pred_scores
            ):
                bboxes_data = []
                for (
                    pred_image_bbox,
                    pred_bbox,
                    pred_detection_score,
                ) in zip(img_boxes, pred_bboxes, pred_scores):
                    ymin, xmin, ymax, xmax = pred_bbox
                    bboxes_data.append(BboxData(
                        image_path=image_data.image_path,
                        image_bbox=pred_image_bbox,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        detection_score=pred_detection_score
                    ))
                images_data.append(ImageData(
                    image_path=image_data.image_path,
                    bboxes_data=bboxes_data
                ))

        return images_data
