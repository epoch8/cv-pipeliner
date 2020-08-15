import copy
from typing import List
from IPython.display import display
from PIL import Image

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.core.visualizer import Visualizer
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.metrics_counters.core.pipeline import get_df_pipeline_matchings
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer
from two_stage_pipeliner.utils.jupyter_visualizer import JupyterVisualizer
from two_stage_pipeliner.visualizers.core.images_data import visualize_image_data, \
    visualize_images_data_side_by_side


class PipelineVisualizer(Visualizer):
    def __init__(self, inferencer: PipelineInferencer = None):
        Visualizer.__init__(self, inferencer)
        self.jupyter_visualizer = None

    def _get_images_data_info_with_TP_FP_FN(
        self,
        true_images_data: List[ImageData],
        pred_images_data: List[ImageData],
        minimum_iou: float
    ):

        true_images_data_info = copy.deepcopy(true_images_data)
        pred_images_data_info = copy.deepcopy(pred_images_data)

        n_true_bboxes = [
            [(bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax)
             for bbox_data in image_data.bboxes_data]
            for image_data in true_images_data_info
        ]
        n_pred_bboxes = [
            [(bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax)
             for bbox_data in image_data.bboxes_data]
            for image_data in pred_images_data_info
        ]
        n_true_labels = [
            [bbox_data.label
             for bbox_data in image_data.bboxes_data]
            for image_data in true_images_data_info
        ]
        n_pred_labels = [
            [bbox_data.label
             for bbox_data in image_data.bboxes_data]
            for image_data in pred_images_data_info
        ]
        df_pipeline_matchings = get_df_pipeline_matchings(
            n_true_bboxes=n_true_bboxes,
            n_pred_bboxes=n_pred_bboxes,
            n_true_labels=n_true_labels,
            n_pred_labels=n_pred_labels,
            minimum_iou=minimum_iou
        )
        for tag, images_data in [('true', true_images_data_info),
                                 ('pred', pred_images_data_info)]:
            for idx, image_data in enumerate(images_data):
                image_data.pipeline_TP = df_pipeline_matchings.loc[idx, 'TP']
                image_data.pipeline_FP = df_pipeline_matchings.loc[idx, 'FP']
                image_data.pipeline_FN = df_pipeline_matchings.loc[idx, 'FN']
                items = df_pipeline_matchings.loc[idx, 'items']
                bboxes = [item[f'{tag}_bbox'] for item in items]
                for bbox_data in image_data.bboxes_data:
                    bbox = (bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax)
                    item_idx = bboxes.index(bbox)
                    error_type = items[item_idx]['error_type']
                    bbox_data.label = f"{bbox_data.label} [{error_type}]"

        return true_images_data_info, pred_images_data_info

    def visualize(self,
                  images_data: List[ImageData],
                  detection_score_threshold: float = None,
                  show_TP_FP_FN: bool = False,
                  minimum_iou: float = None):

        images_data = copy.deepcopy(images_data)

        if self.inferencer is not None and show_TP_FP_FN and minimum_iou is not None:
            images_data_gen = BatchGeneratorImageData(images_data, batch_size=len(images_data))
            pred_images_data = self.inferencer.predict(images_data_gen, detection_score_threshold)
            images_data_info, _ = self._get_images_data_info_with_TP_FP_FN(
                images_data,
                pred_images_data,
                minimum_iou
            )
            images_names = [
                f"{image_data.image_path.name} "
                f"[TP: {image_data.pipeline_TP}, "
                f"FP: {image_data.pipeline_FP}, "
                f"FN: {image_data.pipeline_FN}]"
                for image_data in images_data_info
            ]
        else:
            images_names = [image_data.image_path.name for image_data in images_data]
        images_data_gen = BatchGeneratorImageData(images_data, batch_size=1)
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
                    use_labels=False,
                    score_type=None
                )))
            else:
                if show_TP_FP_FN and minimum_iou is not None:
                    (true_images_data_info,
                     pred_images_data_info) = self._get_images_data_info_with_TP_FP_FN(
                        [self.true_image_data],
                        [self.pred_image_data],
                        minimum_iou=minimum_iou
                    )
                    true_image_data_info = true_images_data_info[0]
                    pred_image_data_info = pred_images_data_info[0]
                    true_labels_info = [bbox_data.label for bbox_data in true_image_data_info.bboxes_data]
                    pred_labels_info = [bbox_data.label for bbox_data in pred_image_data_info.bboxes_data]
                    tags1 = self.jupyter_visualizer.choices.value.split('+')
                    tags2 = self.jupyter_visualizer.choices2.value.split('+')
                    filter_by1 = [
                        value for value in true_labels_info
                        if any(f"[{tag}]" in value for tag in tags1)
                    ]
                    filter_by2 = [
                        value for value in pred_labels_info
                        if any(f"[{tag}]" in value for tag in tags2)
                    ]
                else:
                    true_image_data_info = self.true_image_data
                    pred_image_data_info = self.pred_image_data
                    filter_by1, filter_by2 = None, None

                display(Image.fromarray(visualize_images_data_side_by_side(
                    true_image_data_info, pred_image_data_info,
                    use_labels1=True, use_labels2=True,
                    score_type1=None, score_type2=None,
                    filter_by1=filter_by1, filter_by2=filter_by2
                )))

        if self.jupyter_visualizer is not None:
            del self.jupyter_visualizer
        self.jupyter_visualizer = JupyterVisualizer(
            images=range(len(images_data_gen)),
            images_names=images_names,
            display_fn=display_fn,
            choices=(
                ['TP+FP+FN', 'FP+FN', 'TP', 'FP', 'FN']
                if self.inferencer is not None and show_TP_FP_FN else []
            ),
            choices_description='GT',
            choices2=(
                ['TP+FP', 'TP', 'FP']
                if self.inferencer is not None and show_TP_FP_FN else []
            ),
            choices2_description='Prediction',
        )
        self.jupyter_visualizer.visualize()
