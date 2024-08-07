from typing import Callable, List, Tuple, Union

from tqdm import tqdm

from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.core.inferencer import Inferencer
from cv_pipeliner.inference_models.detection.core import DetectionModel


class DetectionInferencer(Inferencer):
    def __init__(self, model: DetectionModel):
        assert isinstance(model, DetectionModel)
        super().__init__(model)

    def _postprocess_predictions(
        self,
        images_data: List[ImageData],
        n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
        n_k_pred_keypoints: List[List[List[Tuple[int, int]]]],
        n_pred_scores: List[List[float]],
        open_images_in_images_data: bool,
        open_cropped_images_in_bboxes_data: bool,
    ) -> List[ImageData]:
        pred_images_data = []
        for image_data, pred_bboxes, k_pred_keypoints, pred_scores in zip(
            images_data, n_pred_bboxes, n_k_pred_keypoints, n_pred_scores
        ):
            bboxes_data = []
            for pred_bbox, pred_keypoints, pred_detection_score in zip(pred_bboxes, k_pred_keypoints, pred_scores):
                xmin, ymin, xmax, ymax = pred_bbox
                bboxes_data.append(
                    BboxData(
                        image_path=image_data.image_path,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        keypoints=pred_keypoints,
                        detection_score=pred_detection_score,
                    )
                )
            if open_cropped_images_in_bboxes_data:
                for bbox_data in bboxes_data:
                    bbox_data.open_cropped_image(source_image=image_data.image, inplace=True)
            image = image_data.image if open_images_in_images_data else None
            pred_images_data.append(
                ImageData(
                    image_path=image_data.image_path,
                    image=image,
                    bboxes_data=bboxes_data,
                    label=image_data.label,
                    keypoints=image_data.keypoints,
                    additional_info=image_data.additional_info,
                    meta_width=image_data.meta_width,
                    meta_height=image_data.meta_height,
                )
            )

        return pred_images_data

    def predict(
        self,
        images_data_gen: Union[List[ImageData], BatchGeneratorImageData],
        score_threshold: float,
        open_images_in_images_data: bool = False,  # Warning: hard memory use
        open_cropped_images_in_bboxes_data: bool = False,
        disable_tqdm: bool = False,
        progress_callback: Callable[[int], None] = lambda progress: None,
        batch_size_default: int = 16,
    ) -> List[ImageData]:
        if isinstance(images_data_gen, list):
            images_data_gen = BatchGeneratorImageData(images_data_gen, batch_size=batch_size_default)
        assert isinstance(images_data_gen, BatchGeneratorImageData)
        pred_images_data = []
        with tqdm(total=len(images_data_gen.data), disable=disable_tqdm) as pbar:
            for images_data in images_data_gen:
                input = [image_data.image for image_data in images_data]
                n_pred_bboxes, n_k_pred_keypoints, n_pred_scores, _, _ = self.model.predict(
                    input=input, score_threshold=score_threshold
                )
                pred_images_data_batch = self._postprocess_predictions(
                    images_data=images_data,
                    n_pred_bboxes=n_pred_bboxes,
                    n_k_pred_keypoints=n_k_pred_keypoints,
                    n_pred_scores=n_pred_scores,
                    open_images_in_images_data=open_images_in_images_data,
                    open_cropped_images_in_bboxes_data=open_cropped_images_in_bboxes_data,
                )
                pred_images_data.extend(pred_images_data_batch)
                pbar.update(len(images_data))
                progress_callback(pbar.n)

        return pred_images_data
