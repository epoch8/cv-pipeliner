from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData

from cv_pipeliner.core.inferencer import Inferencer
from cv_pipeliner.inference_models.pipeline import PipelineModel


class PipelineInferencer(Inferencer):
    def __init__(self, model: PipelineModel):
        assert isinstance(model, PipelineModel)
        super().__init__(model)

    def _postprocess_predictions(
        self,
        images_data: List[ImageData],
        n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
        n_pred_detection_scores: List[List[float]],
        n_pred_labels_top_n: List[List[List[str]]],
        n_pred_classification_scores_top_n: List[List[List[float]]],
        open_images_in_images_data: bool,
        open_cropped_images_in_bboxes_data: bool
    ) -> List[ImageData]:
        pred_images_data = []
        for (image_data, pred_bboxes,
             pred_detection_scores, pred_labels_top_n, pred_classification_scores_top_n) in zip(
            images_data, n_pred_bboxes,
            n_pred_detection_scores, n_pred_labels_top_n,
            n_pred_classification_scores_top_n
        ):
            bboxes_data = []
            for (
                pred_bbox, pred_detection_score,
                pred_label_top_n, pred_classification_score_top_n
            ) in zip(
                pred_bboxes, pred_detection_scores,
                pred_labels_top_n, pred_classification_scores_top_n
            ):
                xmin, ymin, xmax, ymax = pred_bbox
                bboxes_data.append(BboxData(
                    image_path=image_data.image_path,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    detection_score=pred_detection_score,
                    label=pred_label_top_n[0],
                    classification_score=pred_classification_score_top_n[0],
                    top_n=len(pred_label_top_n),
                    labels_top_n=pred_label_top_n,
                    classification_scores_top_n=pred_classification_score_top_n
                ))
            if open_cropped_images_in_bboxes_data:
                for bbox_data in bboxes_data:
                    bbox_data.open_cropped_image(source_image=image_data.image, inplace=True)
            image = image_data.image if open_images_in_images_data else None
            pred_images_data.append(ImageData(
                image_path=image_data.image_path,
                image=image,
                bboxes_data=bboxes_data
            ))
        return pred_images_data

    def predict(
        self,
        images_data_gen: BatchGeneratorImageData,
        detection_score_threshold: float,
        classification_top_n: int = 1,
        open_images_in_images_data: bool = False,  # Warning: hard memory use
        open_cropped_images_in_bboxes_data: bool = False,
        disable_tqdm: bool = False,
        classification_batch_size: int = 16
    ) -> List[ImageData]:
        assert isinstance(images_data_gen, BatchGeneratorImageData)
        pred_images_data = []
        with tqdm(total=len(images_data_gen.data), disable=disable_tqdm) as pbar:
            for images_data in images_data_gen:
                input = [image_data.image for image_data in images_data]
                input = self.model.preprocess_input(input)
                (
                    n_pred_bboxes,
                    n_pred_detection_scores,
                    n_pred_labels_top_n,
                    n_pred_classification_scores_top_n
                ) = self.model.predict(
                    input=input,
                    detection_score_threshold=detection_score_threshold,
                    classification_top_n=classification_top_n,
                    classification_batch_size=classification_batch_size
                )
                pred_images_data_batch = self._postprocess_predictions(
                    images_data=images_data,
                    n_pred_bboxes=n_pred_bboxes,
                    n_pred_detection_scores=n_pred_detection_scores,
                    n_pred_labels_top_n=n_pred_labels_top_n,
                    n_pred_classification_scores_top_n=n_pred_classification_scores_top_n,
                    open_images_in_images_data=open_images_in_images_data,
                    open_cropped_images_in_bboxes_data=open_cropped_images_in_bboxes_data
                )
                pred_images_data.extend(pred_images_data_batch)
                pbar.update(len(images_data))

        return pred_images_data

    @property
    def class_names(self):
        return self.model.class_names
