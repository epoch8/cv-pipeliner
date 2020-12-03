import copy
from typing import List
from IPython.display import display
from PIL import Image

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.core.visualizer import Visualizer
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.inferencers.pipeline import PipelineInferencer
from cv_pipeliner.utils.jupyter_visualizer import JupyterVisualizer
from cv_pipeliner.visualizers.core.image_data import visualize_image_data, \
    visualize_images_data_side_by_side
from cv_pipeliner.metrics.image_data_matching import ImageDataMatching
from cv_pipeliner.visualizers.core.image_data_matching import visualize_image_data_matching_side_by_side


class PipelineVisualizer(Visualizer):
    def __init__(self, inferencer: PipelineInferencer = None):
        if inferencer is not None:
            assert isinstance(inferencer, PipelineInferencer)
        super().__init__(inferencer)
        self.jupyter_visualizer = None

    def _get_images_names_by_inference(
        self,
        images_data: List[ImageData],
        detection_score_threshold: float,
        minimum_iou: float,
        extra_bbox_label: str,
        batch_size: int
    ) -> List[str]:
        images_data_gen = BatchGeneratorImageData(images_data,
                                                  batch_size=batch_size,
                                                  use_not_caught_elements_as_last_batch=True)
        pred_images_data = self.inferencer.predict(images_data_gen, detection_score_threshold)
        images_data_matchings = [
            ImageDataMatching(
                true_image_data=image_data,
                pred_image_data=pred_image_data,
                minimum_iou=minimum_iou,
                extra_bbox_label=extra_bbox_label
            )
            for image_data, pred_image_data in zip(images_data, pred_images_data)
        ]
        images_names = [
            f"{image_data_matching.true_image_data.image_name} "
            f"[TP: {image_data_matching.get_pipeline_TP()}, "
            f"FP: {image_data_matching.get_pipeline_FP()}, "
            f"FN: {image_data_matching.get_pipeline_FN()}, "
            f"TP (extra bbox): {image_data_matching.get_pipeline_TP_extra_bbox()}, "
            f"FP (extra bbox): {image_data_matching.get_pipeline_FP_extra_bbox()}]"
            for image_data_matching in images_data_matchings
        ]
        return images_names

    def visualize(self,
                  images_data: List[ImageData],
                  detection_score_threshold: float = None,
                  show_TP_FP_FN_with_minimum_iou: float = None,
                  extra_bbox_label: str = None,
                  batch_size: int = 16):

        images_data = copy.deepcopy(images_data)

        if self.inferencer is not None and show_TP_FP_FN_with_minimum_iou is not None:
            images_names = self._get_images_names_by_inference(
                images_data=images_data,
                detection_score_threshold=detection_score_threshold,
                minimum_iou=show_TP_FP_FN_with_minimum_iou,
                extra_bbox_label=extra_bbox_label,
                batch_size=batch_size
            )
        else:
            images_names = [image_data.image_name for image_data in images_data]
        images_data_gen = BatchGeneratorImageData(images_data, batch_size=1,
                                                  use_not_caught_elements_as_last_batch=True)
        self.i = None

        def display_fn(i):
            if self.i is None or i != self.i:
                self.true_image_data = images_data_gen[i][0]
                if self.inferencer is not None:
                    image_data_gen = BatchGeneratorImageData([self.true_image_data], batch_size=1,
                                                             use_not_caught_elements_as_last_batch=True)
                    self.pred_image_data = self.inferencer.predict(image_data_gen,
                                                                   detection_score_threshold)[0]
                self.i = i

            if self.inferencer is None:
                display(Image.fromarray(visualize_image_data(
                    self.true_image_data,
                    use_labels=True,
                    score_type=None
                )))
            else:
                if show_TP_FP_FN_with_minimum_iou is not None:
                    display(Image.fromarray(visualize_image_data_matching_side_by_side(
                        image_data_matching=ImageDataMatching(
                            true_image_data=self.true_image_data,
                            pred_image_data=self.pred_image_data,
                            minimum_iou=show_TP_FP_FN_with_minimum_iou,
                            extra_bbox_label=extra_bbox_label
                        ),
                        error_type='pipeline',
                        true_use_labels=True, pred_use_labels=True,
                        true_score_type=None, pred_score_type=None,
                        true_filter_by_error_types=self.jupyter_visualizer.choices.value.split('+'),
                        pred_filter_by_error_types=self.jupyter_visualizer.choices2.value.split('+')
                    )))
                else:
                    display(Image.fromarray(visualize_images_data_side_by_side(
                        image_data1=self.true_image_data,
                        image_data2=self.pred_image_data,
                        use_labels1=True, use_labels2=True,
                        score_type1=None, score_type2=None
                    )))

        self.jupyter_visualizer = JupyterVisualizer(
            images=range(len(images_data_gen)),
            images_names=images_names,
            display_fn=display_fn,
            choices=(
                [
                    'TP+FP+FN', 'FP+FN',
                    'TP', 'FP', 'FN'
                ]
                if self.inferencer is not None and show_TP_FP_FN_with_minimum_iou is not None else []
            ),
            choices_description='GT',
            choices2=(
                [
                    'TP+TP (extra bbox)+FP+FP (extra bbox)+FN (extra bbox)',
                    'FP+FP (extra bbox)+FN (extra bbox)',
                    'TP', 'TP (extra bbox)', 'FP', 'FP (extra bbox)', 'FN (extra bbox)'
                ]
                if self.inferencer is not None and show_TP_FP_FN_with_minimum_iou is not None else []
            ),
            choices2_description='Prediction',
        )
        self.jupyter_visualizer.visualize()
