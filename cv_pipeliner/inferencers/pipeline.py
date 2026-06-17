from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inferencers.base import Inferencer, ModelSpec
from cv_pipeliner.inferencers.batch_utils import call_progress_callback, ensure_image_data_generator
from cv_pipeliner.inferencers.classification import ClassificationInferencer
from cv_pipeliner.inferencers.classification.core import ClassificationModelSpec
from cv_pipeliner.inferencers.detection import DetectionInferencer
from cv_pipeliner.inferencers.detection.core import DetectionModelSpec
from cv_pipeliner.inferencers.postprocess import build_detection_images_data
from cv_pipeliner.inferencers.results import DetectionResult
from cv_pipeliner.logging import logger


class PipelineModelSpec(ModelSpec):
    detection_model_spec: DetectionModelSpec
    classification_model_spec: Union[ClassificationModelSpec, None] = None

    @property
    def runtime_cls(self):
        raise NotImplementedError("PipelineModelSpec composes inferencers and does not have a runtime.")

    def load_runtime(self, **kwargs):
        raise NotImplementedError("PipelineModelSpec composes inferencers and does not have a runtime.")

    def load(self, **kwargs):
        return self.load_pipeline_inferencer(**kwargs)

    def load_pipeline_inferencer(self) -> "PipelineInferencer":
        detection_inferencer = self.detection_model_spec.load_detection_inferencer()
        classification_inferencer = (
            self.classification_model_spec.load_classification_inferencer()
            if self.classification_model_spec is not None
            else None
        )
        return PipelineInferencer(
            detection_inferencer=detection_inferencer,
            classification_inferencer=classification_inferencer,
            model_spec=self,
        )


class PipelineInferencer(Inferencer):
    def __init__(
        self,
        detection_inferencer: DetectionInferencer = None,
        classification_inferencer: ClassificationInferencer = None,
        model_spec: PipelineModelSpec = None,
    ):
        assert isinstance(detection_inferencer, DetectionInferencer)
        self.runtime = None
        self.model = None
        self.model_spec = model_spec
        self.detection_inferencer = detection_inferencer
        self.classification_inferencer = classification_inferencer
        self.detection_model = detection_inferencer.runtime
        self.classification_model = classification_inferencer.runtime if classification_inferencer is not None else None

    def _postprocess_predictions(
        self,
        images_data: List[ImageData],
        n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
        n_k_pred_keypoints: List[List[Tuple[int, int]]],
        n_k_pred_masks: List[List[List[List[Tuple[int, int]]]]],
        n_pred_detection_scores: List[List[float]],
        n_pred_labels_top_n: List[List[List[str]]],
        n_pred_classification_scores_top_n: List[List[List[float]]],
        open_images_in_images_data: bool,
        open_cropped_images_in_bboxes_data: bool,
    ) -> List[ImageData]:
        return build_detection_images_data(
            images_data=images_data,
            detection_result=DetectionResult(
                bboxes=n_pred_bboxes,
                keypoints=n_k_pred_keypoints,
                masks=n_k_pred_masks,
                detection_scores=n_pred_detection_scores,
                labels_top_n=n_pred_labels_top_n,
                classification_scores_top_n=n_pred_classification_scores_top_n,
            ),
            open_images_in_images_data=open_images_in_images_data,
            open_cropped_images_in_bboxes_data=open_cropped_images_in_bboxes_data,
        )

    def _predict_batch(
        self,
        images_data: List[ImageData],
        detection_score_threshold: float,
        classification_top_n: int,
        classification_batch_size: int,
        detection_kwargs: Dict[str, Any],
        classification_kwargs: Dict[str, Any],
        disable_tqdm_classification: bool,
        open_images_in_images_data: bool,
        open_cropped_images_in_bboxes_data: bool,
    ) -> List[ImageData]:
        input = [image_data.image for image_data in images_data]
        logger.debug("Running detection...")
        detection_result = DetectionResult.from_tuple(
            self.detection_model.predict(
                input,
                score_threshold=detection_score_threshold,
                classification_top_n=classification_top_n,
                **detection_kwargs,
            )
        )
        logger.debug(f"Detection: found {np.sum([len(pred_bboxes) for pred_bboxes in detection_result.bboxes])} bboxes!")

        pred_images_data = build_detection_images_data(
            images_data=images_data,
            detection_result=detection_result,
            open_images_in_images_data=True,
            open_cropped_images_in_bboxes_data=self.classification_inferencer is not None,
        )

        if self.classification_inferencer is None:
            if not open_images_in_images_data:
                for image_data in pred_images_data:
                    image_data.image = None
            if open_cropped_images_in_bboxes_data and detection_result.labels_top_n is not None:
                for pred_image_data, source_image_data in zip(pred_images_data, images_data):
                    for bbox_data in pred_image_data.bboxes_data:
                        bbox_data.open_cropped_image(source_image=source_image_data.image, inplace=True)
            return pred_images_data

        logger.debug("Running classification...")
        n_bboxes_data = [image_data.bboxes_data for image_data in pred_images_data]
        bboxes_data_gen = BatchGeneratorBboxData(n_bboxes_data, batch_size=classification_batch_size)
        n_pred_bboxes_data = self.classification_inferencer.predict(
            bboxes_data_gen,
            top_n=classification_top_n,
            open_images_in_data=open_cropped_images_in_bboxes_data,
            disable_tqdm=disable_tqdm_classification,
            model_kwargs=classification_kwargs,
        )
        for pred_image_data, bboxes_data in zip(pred_images_data, n_pred_bboxes_data):
            pred_image_data.bboxes_data = bboxes_data
            if not open_images_in_images_data:
                pred_image_data.image = None
        logger.debug("Classification end!")
        return pred_images_data

    def predict(
        self,
        images_data_gen: Union[List[ImageData], BatchGeneratorImageData],
        detection_score_threshold: float,
        classification_top_n: int = 1,
        open_images_in_images_data: bool = False,  # Warning: hard memory use
        open_cropped_images_in_bboxes_data: bool = False,
        disable_tqdm: bool = False,
        disable_tqdm_classification: bool = False,
        classification_batch_size: int = 16,
        detection_kwargs: Dict[str, Any] = None,
        classification_kwargs: Dict[str, Any] = None,
        progress_callback: Callable[[int], None] = lambda progress: None,
        batch_size_default: int = 16,
    ) -> List[ImageData]:
        detection_kwargs = detection_kwargs or {}
        classification_kwargs = classification_kwargs or {}
        images_data_gen = ensure_image_data_generator(images_data_gen, batch_size_default=batch_size_default)
        pred_images_data = []
        progress = 0
        with tqdm(total=len(images_data_gen.data), disable=disable_tqdm) as pbar:
            for images_data in images_data_gen:
                pred_images_data_batch = self._predict_batch(
                    images_data=images_data,
                    detection_score_threshold=detection_score_threshold,
                    classification_top_n=classification_top_n,
                    classification_batch_size=classification_batch_size,
                    detection_kwargs=detection_kwargs,
                    classification_kwargs=classification_kwargs,
                    disable_tqdm_classification=disable_tqdm_classification,
                    open_images_in_images_data=open_images_in_images_data,
                    open_cropped_images_in_bboxes_data=open_cropped_images_in_bboxes_data,
                )
                pred_images_data.extend(pred_images_data_batch)
                pbar.update(len(images_data))
                progress += len(images_data)
                call_progress_callback(progress_callback, progress)

        return pred_images_data

    @property
    def class_names(self):
        if self.classification_inferencer is not None:
            return self.classification_inferencer.class_names
        return self.detection_model.class_names
