import copy
from typing import List
from IPython.display import display

from two_stage_pipeliner.metrics_counters.core.detection import get_df_detector_matchings
from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.core.visualizer import Visualizer
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.utils.jupyter_visualizer import JupyterVisualizer
from two_stage_pipeliner.visualizers.core.images_data import visualize_image_data, \
    visualize_images_data_side_by_side


class DetectionVisualizer(Visualizer):
    def __init__(self, inferencer: DetectionInferencer = None):
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
        df_detector_matchings = get_df_detector_matchings(
            n_true_bboxes=n_true_bboxes,
            n_pred_bboxes=n_pred_bboxes,
            minimum_iou=minimum_iou
        )
        for tag, images_data in [('true', true_images_data_info),
                                 ('pred', pred_images_data_info)]:
            for idx, image_data in enumerate(images_data):
                image_data.detector_TP = df_detector_matchings.loc[idx, 'TP']
                image_data.detector_FP = df_detector_matchings.loc[idx, 'FP']
                image_data.detector_FN = df_detector_matchings.loc[idx, 'FN']
                items = df_detector_matchings.loc[idx, 'items']
                bboxes = [item[f'{tag}_bbox'] for item in items]
                for bbox_data in image_data.bboxes_data:
                    bbox = (bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax)
                    if bbox in bboxes:
                        item_idx = bboxes.index(bbox)
                        if items[item_idx]['found']:
                            bbox_data.label = "TP"
                        else:
                            bbox_data.label = "FN"
                    else:
                        bbox_data.label = "FP"

        return true_images_data_info, pred_images_data_info

    def visualize(self,
                  images_data: List[ImageData],
                  score_threshold: float = None,
                  show_TP_FP_FN: bool = False,
                  minimum_iou: float = None):

        images_data = copy.deepcopy(images_data)

        if self.inferencer is not None and show_TP_FP_FN and minimum_iou is not None:
            images_data_gen = BatchGeneratorImageData(images_data, batch_size=len(images_data))
            pred_images_data = self.inferencer.predict(images_data_gen, score_threshold)
            images_data_info, _ = self._get_images_data_info_with_TP_FP_FN(
                images_data,
                pred_images_data,
                minimum_iou
            )
            images_names = [
                f"{image_data.image_path.name} "
                f"[TP: {image_data.detector_TP}, "
                f"FP: {image_data.detector_FP}, "
                f"FN: {image_data.detector_FN}]"
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
                    self.pred_image_data = self.inferencer.predict([self.batch], score_threshold)[0]
                self.i = i

            if self.inferencer is None:
                display(visualize_image_data(
                    self.true_image_data,
                    use_labels=False,
                    score_type=None
                ))
            else:
                if show_TP_FP_FN and minimum_iou is not None:
                    true_images_data_info, pred_images_data_info = self._get_images_data_info_with_TP_FP_FN(
                        [self.true_image_data],
                        [self.pred_image_data],
                        minimum_iou=minimum_iou
                    )
                    true_image_data_info = true_images_data_info[0]
                    pred_image_data_info = pred_images_data_info[0]
                    use_labels = True
                    filter_by1 = self.jupyter_visualizer.choices.value.split('+')
                    filter_by2 = self.jupyter_visualizer.choices2.value.split('+')
                else:
                    true_image_data_info = self.true_image_data
                    pred_image_data_info = self.pred_image_data
                    use_labels = False
                    filter_by1, filter_by2 = None, None

                display(visualize_images_data_side_by_side(
                    true_image_data_info, pred_image_data_info,
                    use_labels1=use_labels, use_labels2=use_labels,
                    score_type1=None, score_type2='detection',
                    filter_by1=filter_by1, filter_by2=filter_by2
                ))

        if self.jupyter_visualizer is not None:
            del self.jupyter_visualizer
        self.jupyter_visualizer = JupyterVisualizer(
            images=range(len(images_data_gen)),
            images_names=images_names,
            display_fn=display_fn,
            choices=(
                ['TP+FN', 'TP', 'FN']
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
