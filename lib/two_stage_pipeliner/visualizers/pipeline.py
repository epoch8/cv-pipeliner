import copy
from typing import List
from IPython.display import display
from PIL import Image

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.core.visualizer import Visualizer
from two_stage_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer
from two_stage_pipeliner.utils.jupyter_visualizer import JupyterVisualizer
from two_stage_pipeliner.visualizers.core.image_data import visualize_image_data, \
    visualize_images_data_side_by_side
from two_stage_pipeliner.metrics.image_data_matching import ImageDataMatching
from two_stage_pipeliner.visualizers.core.image_data_matching import visualize_image_data_matching_side_by_side


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
        use_soft_metrics_with_known_labels: List[str]
    ) -> List[str]:
        images_data_gen = BatchGeneratorImageData(images_data,
                                                  batch_size=min(len(images_data), 16),
                                                  use_not_caught_elements_as_last_batch=True)
        pred_images_data = self.inferencer.predict(images_data_gen, detection_score_threshold)
        images_data_matchings = [
            ImageDataMatching(image_data, pred_image_data, minimum_iou)
            for image_data, pred_image_data in zip(images_data, pred_images_data)
        ]
        images_names = []
        for image_data_matching in images_data_matchings:
            TP = image_data_matching.get_pipeline_TP(
                extra_bbox_label=extra_bbox_label,
                use_soft_metrics_with_known_labels=use_soft_metrics_with_known_labels
            )
            FP = image_data_matching.get_pipeline_FP(
                extra_bbox_label=extra_bbox_label,
                use_soft_metrics_with_known_labels=use_soft_metrics_with_known_labels
            )
            FN = image_data_matching.get_pipeline_FN(
                extra_bbox_label=extra_bbox_label,
                use_soft_metrics_with_known_labels=use_soft_metrics_with_known_labels
            )
            TP_extra_bbox = image_data_matching.get_pipeline_TP_extra_bbox(
                extra_bbox_label=extra_bbox_label,
            )
            FN_extra_bbox = image_data_matching.get_pipeline_FP_extra_bbox(
                extra_bbox_label=extra_bbox_label,
            )
            images_names.append(
                f"{image_data_matching.true_image_data.image_path.name} "
                f"[TP: {TP}, "
                f"FP: {FP}, "
                f"FN: {FN}, "
                f"TP (extra bbox): {TP_extra_bbox}, "
                f"FP (extra bbox): {FN_extra_bbox}]"
                for image_data_matching in images_data_matchings
            )
        return images_names

    def visualize(self,
                  images_data: List[ImageData],
                  detection_score_threshold: float = None,
                  show_TP_FP_FN_with_minimum_iou: float = None,
                  extra_bbox_label: str = None,
                  use_soft_metrics_with_known_labels: List[str] = None):

        images_data = copy.deepcopy(images_data)

        if self.inferencer is not None and show_TP_FP_FN_with_minimum_iou is not None:
            images_names = self._get_images_names_by_inference(
                images_data=images_data,
                detection_score_threshold=detection_score_threshold,
                minimum_iou=show_TP_FP_FN_with_minimum_iou,
                extra_bbox_label=extra_bbox_label,
                use_soft_metrics_with_known_labels=use_soft_metrics_with_known_labels
            )
        else:
            images_names = [image_data.image_path.name for image_data in images_data]
        images_data_gen = BatchGeneratorImageData(images_data, batch_size=1,
                                                  use_not_caught_elements_as_last_batch=True)
        self.i = None

        def display_fn(i):
            if self.i is None or i != self.i:
                self.batch = images_data_gen[i]
                self.true_image_data = self.batch[0]
                if self.inferencer is not None:
                    self.pred_image_data = self.inferencer.predict([self.batch],
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
                        image_data_matching=ImageDataMatching(self.true_image_data, self.pred_image_data,
                                                              show_TP_FP_FN_with_minimum_iou),
                        error_type='pipeline',
                        true_use_labels=True, pred_use_labels=True,
                        true_score_type=None, pred_score_type=None,
                        true_filter_by_error_types=self.jupyter_visualizer.choices.value.split('+'),
                        pred_filter_by_error_types=self.jupyter_visualizer.choices2.value.split('+'),
                        extra_bbox_label=extra_bbox_label,
                        use_soft_metrics_with_known_labels=use_soft_metrics_with_known_labels
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
                    'TP+TP (extra bbox)+FP+FP (extra bbox)',
                    'FP+FP (extra bbox)',
                    'TP', 'TP (extra bbox)', 'FP', 'FP (extra bbox)'
                ]
                if self.inferencer is not None and show_TP_FP_FN_with_minimum_iou is not None else []
            ),
            choices2_description='Prediction',
        )
        self.jupyter_visualizer.visualize()
