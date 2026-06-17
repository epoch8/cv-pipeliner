import abc
from typing import Callable, List, Tuple, Type, Union

import numpy as np
from tqdm import tqdm

from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.data import ImageData
from cv_pipeliner.inferencers.base import Inferencer, ModelSpec, Runtime
from cv_pipeliner.inferencers.batch_utils import call_progress_callback, ensure_image_data_generator
from cv_pipeliner.inferencers.postprocess import build_detection_images_data
from cv_pipeliner.inferencers.results import DetectionResult

Bbox = Tuple[int, int, int, int]
Score = float
Class = str

Bboxes = List[Bbox]
DetectionScores = List[Score]
ClassificationScores = List[Score]
Classes = List[Class]
Keypoints = List[List[Tuple[int, int]]]
Mask = List[List[List[Tuple[int, int]]]]

DetectionInput = List[np.ndarray]
DetectionOutput = Tuple[
    List[Bboxes],
    List[Keypoints],
    List[Mask],
    List[DetectionScores],
    List[Classes],
    List[ClassificationScores],
]


class DetectionModelSpec(ModelSpec):
    class_names: List[str] = None

    @property
    @abc.abstractmethod
    def runtime_cls(self) -> Type["DetectionRuntime"]:
        pass

    def load_detection_inferencer(self) -> "DetectionInferencer":
        return DetectionInferencer(self.load_runtime())


class DetectionRuntime(Runtime):
    def __init__(self, model_spec: DetectionModelSpec):
        assert isinstance(model_spec, DetectionModelSpec)
        super().__init__(model_spec=model_spec)

    @abc.abstractmethod
    def predict(
        self, input: DetectionInput, score_threshold: float, classification_top_n: int = None
    ) -> DetectionOutput:
        pass


class DetectionInferencer(Inferencer):
    def __init__(self, runtime: DetectionRuntime):
        assert isinstance(runtime, DetectionRuntime)
        super().__init__(runtime)

    def _postprocess_predictions(
        self,
        images_data: List[ImageData],
        n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
        n_k_pred_keypoints: List[List[List[Tuple[int, int]]]],
        n_k_pred_masks: List[List[List[List[Tuple[int, int]]]]],
        n_pred_scores: List[List[float]],
        open_images_in_images_data: bool,
        open_cropped_images_in_bboxes_data: bool,
    ) -> List[ImageData]:
        return build_detection_images_data(
            images_data=images_data,
            detection_result=DetectionResult(
                bboxes=n_pred_bboxes,
                keypoints=n_k_pred_keypoints,
                masks=n_k_pred_masks,
                detection_scores=n_pred_scores,
            ),
            open_images_in_images_data=open_images_in_images_data,
            open_cropped_images_in_bboxes_data=open_cropped_images_in_bboxes_data,
        )

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
        images_data_gen = ensure_image_data_generator(images_data_gen, batch_size_default=batch_size_default)
        pred_images_data = []
        progress = 0
        with tqdm(total=len(images_data_gen.data), disable=disable_tqdm) as pbar:
            for images_data in images_data_gen:
                input = [image_data.image for image_data in images_data]
                n_pred_bboxes, n_k_pred_keypoints, n_k_pred_masks, n_pred_scores, _, _ = self.model.predict(
                    input=input, score_threshold=score_threshold
                )
                pred_images_data_batch = self._postprocess_predictions(
                    images_data=images_data,
                    n_pred_bboxes=n_pred_bboxes,
                    n_k_pred_keypoints=n_k_pred_keypoints,
                    n_k_pred_masks=n_k_pred_masks,
                    n_pred_scores=n_pred_scores,
                    open_images_in_images_data=open_images_in_images_data,
                    open_cropped_images_in_bboxes_data=open_cropped_images_in_bboxes_data,
                )
                pred_images_data.extend(pred_images_data_batch)
                pbar.update(len(images_data))
                progress += len(images_data)
                call_progress_callback(progress_callback, progress)

        return pred_images_data

