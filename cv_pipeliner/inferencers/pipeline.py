from typing import Any, Callable, Dict, List, Tuple
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
        n_k_pred_keypoints: List[List[Tuple[int, int]]],
        n_pred_detection_scores: List[List[float]],
        n_pred_labels_top_n: List[List[List[str]]],
        n_pred_classification_scores_top_n: List[List[List[float]]],
        open_images_in_images_data: bool,
        open_cropped_images_in_bboxes_data: bool
    ) -> List[ImageData]:
        pred_images_data = []
        for (image_data, pred_bboxes, k_pred_keypoints,
             pred_detection_scores, pred_labels_top_n, pred_classification_scores_top_n) in zip(
            images_data, n_pred_bboxes, n_k_pred_keypoints,
            n_pred_detection_scores, n_pred_labels_top_n,
            n_pred_classification_scores_top_n
        ):
            bboxes_data = []
            for (
                pred_bbox, pred_keypoints, pred_detection_score,
                pred_label_top_n, pred_classification_score_top_n
            ) in zip(
                pred_bboxes, k_pred_keypoints, pred_detection_scores,
                pred_labels_top_n, pred_classification_scores_top_n
            ):
                xmin, ymin, xmax, ymax = pred_bbox
                bboxes_data.append(BboxData(
                    image_path=image_data.image_path,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    keypoints=pred_keypoints,
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
                bboxes_data=bboxes_data,
                additional_info=image_data.additional_info,
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
        classification_batch_size: int = 16,
        detection_kwargs: Dict[str, Any] = {},
        classification_kwargs: Dict[str, Any] = {},
        progress_callback: Callable[[int], None] = lambda progress: None
    ) -> List[ImageData]:
        assert isinstance(images_data_gen, BatchGeneratorImageData)
        pred_images_data = []
        with tqdm(total=len(images_data_gen.data), disable=disable_tqdm) as pbar:
            for images_data in images_data_gen:
                input = [image_data.image for image_data in images_data]
                (
                    n_pred_bboxes,
                    n_k_pred_keypoints,
                    n_pred_detection_scores,
                    n_pred_labels_top_n,
                    n_pred_classification_scores_top_n
                ) = self.model.predict(
                    input=input,
                    detection_score_threshold=detection_score_threshold,
                    classification_top_n=classification_top_n,
                    classification_batch_size=classification_batch_size,
                    detection_kwargs=detection_kwargs,
                    classification_kwargs=classification_kwargs
                )
                pred_images_data_batch = self._postprocess_predictions(
                    images_data=images_data,
                    n_pred_bboxes=n_pred_bboxes,
                    n_k_pred_keypoints=n_k_pred_keypoints,
                    n_pred_detection_scores=n_pred_detection_scores,
                    n_pred_labels_top_n=n_pred_labels_top_n,
                    n_pred_classification_scores_top_n=n_pred_classification_scores_top_n,
                    open_images_in_images_data=open_images_in_images_data,
                    open_cropped_images_in_bboxes_data=open_cropped_images_in_bboxes_data
                )
                pred_images_data.extend(pred_images_data_batch)
                pbar.update(len(images_data))
                progress_callback(pbar.n)

        return pred_images_data

    @property
    def class_names(self):
        return self.model.class_names
