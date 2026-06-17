import abc
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
from tqdm import tqdm

from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.inferencers.base import Inferencer, ModelSpec, Runtime
from cv_pipeliner.inferencers.batch_utils import (
    call_progress_callback,
    ensure_image_or_bbox_data_generator,
    split_chunks,
)

Label = str
Score = float

Labels_Top_N = List[List[Label]]
Scores_Top_N = List[List[Score]]

ClassificationInput = List[np.ndarray]
ClassificationOutput = Tuple[Labels_Top_N, Scores_Top_N]


class ClassificationModelSpec(ModelSpec):
    @property
    @abc.abstractmethod
    def runtime_cls(self) -> Type["ClassificationRuntime"]:
        pass

    def load_classification_inferencer(self) -> "ClassificationInferencer":
        return ClassificationInferencer(self.load_runtime())


class ClassificationRuntime(Runtime):
    def __init__(self, model_spec: ClassificationModelSpec):
        assert isinstance(model_spec, ClassificationModelSpec)
        super().__init__(model_spec=model_spec)

    @abc.abstractmethod
    def predict(self, input: ClassificationInput, top_n: int = 1) -> ClassificationOutput:
        pass

    @property
    @abc.abstractmethod
    def class_names(self) -> List[str]:
        pass


class ClassificationInferencer(Inferencer):
    def __init__(self, runtime: ClassificationRuntime):
        assert isinstance(runtime, ClassificationRuntime)
        super().__init__(runtime)

    def _postprocess_images_data(
        self,
        images_data: List[ImageData],
        pred_labels_top_n: List[List[str]],
        pred_scores_top_n: List[List[float]],
        open_images_in_images_data: bool,
    ) -> List[ImageData]:
        images_data_res = []
        for image_data, pred_label_top_n, pred_classification_score_top_n in zip(
            images_data, pred_labels_top_n, pred_scores_top_n
        ):
            image = image_data.image if open_images_in_images_data else None
            images_data_res.append(
                ImageData(
                    image_path=image_data.image_path,
                    image=image,
                    label=pred_label_top_n[0],
                    bboxes_data=image_data.bboxes_data,
                    keypoints=image_data.keypoints,
                    classification_score=pred_classification_score_top_n[0],
                    top_n=len(pred_label_top_n),
                    labels_top_n=pred_label_top_n,
                    classification_scores_top_n=pred_classification_score_top_n,
                    additional_info=image_data.additional_info,
                    meta_height=image_data.meta_height,
                    meta_width=image_data.meta_width,
                )
            )

        return images_data_res

    def _predict_images_data(
        self,
        images_data_gen: BatchGeneratorImageData,
        top_n: int = 1,
        open_images_in_images_data: bool = False,
        disable_tqdm: bool = False,
        progress_callback: Callable[[int], None] = lambda progress: None,
        model_kwargs: Dict[str, Any] = None,
    ):
        model_kwargs = model_kwargs or {}
        pred_images_data = []
        count = 0
        with tqdm(total=len(images_data_gen.data), disable=disable_tqdm) as pbar:
            for images_data in images_data_gen:
                input = [image_data.image for image_data in images_data]
                pred_labels_top_n, pred_scores_top_n = self.model.predict(input=input, top_n=top_n, **model_kwargs)
                pred_images_data.extend(
                    self._postprocess_images_data(
                        images_data=images_data,
                        pred_labels_top_n=pred_labels_top_n,
                        pred_scores_top_n=pred_scores_top_n,
                        open_images_in_images_data=open_images_in_images_data,
                    )
                )
                pbar.update(len(images_data))
                count += len(images_data)
                call_progress_callback(progress_callback, count)

        return pred_images_data

    def _postprocess_predictions_bboxes_data(
        self,
        bboxes_data: List[BboxData],
        pred_labels_top_n: List[List[str]],
        pred_scores_top_n: List[List[float]],
        open_cropped_images_in_bboxes_data: bool,
    ) -> List[List[BboxData]]:
        bboxes_data_res = []
        for bbox_data, pred_label_top_n, pred_classification_score_top_n in zip(
            bboxes_data, pred_labels_top_n, pred_scores_top_n
        ):
            cropped_image = bbox_data.cropped_image if open_cropped_images_in_bboxes_data else None
            bboxes_data_res.append(
                BboxData(
                    image_path=bbox_data.image_path,
                    cropped_image=cropped_image,
                    xmin=bbox_data.xmin,
                    ymin=bbox_data.ymin,
                    xmax=bbox_data.xmax,
                    ymax=bbox_data.ymax,
                    detection_score=bbox_data.detection_score,
                    label=pred_label_top_n[0],
                    keypoints=bbox_data.keypoints,
                    classification_score=pred_classification_score_top_n[0],
                    top_n=len(pred_label_top_n),
                    labels_top_n=pred_label_top_n,
                    classification_scores_top_n=pred_classification_score_top_n,
                    additional_info=bbox_data.additional_info,
                    meta_width=bbox_data.meta_width,
                    meta_height=bbox_data.meta_height,
                )
            )

        return bboxes_data_res

    def _split_chunks(self, _list: List, shapes: List[int]) -> List:
        return split_chunks(_list, shapes)

    def _predict_bboxes_data(
        self,
        n_bboxes_data_gen: BatchGeneratorBboxData,
        top_n: int = 1,
        open_images_in_bboxes_data: bool = False,
        disable_tqdm: bool = False,
        progress_callback: Callable[[int], None] = lambda progress: None,
        model_kwargs: Dict[str, Any] = None,
    ):
        model_kwargs = model_kwargs or {}
        pred_bboxes_data = []
        count = 0
        with tqdm(total=len(n_bboxes_data_gen.data), disable=disable_tqdm) as pbar:
            for bboxes_data in n_bboxes_data_gen:
                input = [bbox_data.cropped_image for bbox_data in bboxes_data]
                pred_labels_top_n, pred_scores_top_n = self.model.predict(input=input, top_n=top_n, **model_kwargs)
                pred_bboxes_data.extend(
                    self._postprocess_predictions_bboxes_data(
                        bboxes_data=bboxes_data,
                        pred_labels_top_n=pred_labels_top_n,
                        pred_scores_top_n=pred_scores_top_n,
                        open_cropped_images_in_bboxes_data=open_images_in_bboxes_data,
                    )
                )
                pbar.update(len(bboxes_data))
                count += len(bboxes_data)
                call_progress_callback(progress_callback, count)

        n_pred_bboxes_data = self._split_chunks(pred_bboxes_data, n_bboxes_data_gen.shapes)
        return n_pred_bboxes_data

    def predict(
        self,
        data_gen: Union[BatchGeneratorImageData, BatchGeneratorBboxData],
        top_n: int = 1,
        open_images_in_data: bool = False,
        disable_tqdm: bool = False,
        progress_callback: Callable[[int], None] = lambda progress: None,
        batch_size_default: int = 32,
        model_kwargs: Dict[str, Any] = None,
    ) -> Union[List[ImageData], List[List[BboxData]]]:
        model_kwargs = model_kwargs or {}
        data_gen = ensure_image_or_bbox_data_generator(data_gen, batch_size_default=batch_size_default)
        if isinstance(data_gen, BatchGeneratorImageData):
            return self._predict_images_data(
                images_data_gen=data_gen,
                top_n=top_n,
                open_images_in_images_data=open_images_in_data,
                disable_tqdm=disable_tqdm,
                progress_callback=progress_callback,
                model_kwargs=model_kwargs,
            )
        elif isinstance(data_gen, BatchGeneratorBboxData):
            return self._predict_bboxes_data(
                n_bboxes_data_gen=data_gen,
                top_n=top_n,
                open_images_in_bboxes_data=open_images_in_data,
                disable_tqdm=disable_tqdm,
                progress_callback=progress_callback,
                model_kwargs=model_kwargs,
            )
        else:
            raise TypeError(f"Unknown type of data_gen: {type(data_gen)}")

    @property
    def class_names(self):
        return self.model.class_names

ClassificationRuntime = ClassificationRuntime
