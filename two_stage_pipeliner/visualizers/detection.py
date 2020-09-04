import copy
from typing import List
from IPython.display import display
from PIL import Image

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.core.visualizer import Visualizer
from two_stage_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.utils.jupyter_visualizer import JupyterVisualizer
from two_stage_pipeliner.visualizers.core.image_data import visualize_image_data, \
    visualize_images_data_side_by_side
from two_stage_pipeliner.metrics.image_data_matching import ImageDataMatching
from two_stage_pipeliner.visualizers.core.image_data_matching import visualize_image_data_matching_side_by_side


class DetectionVisualizer(Visualizer):
    def __init__(self, inferencer: DetectionInferencer = None):
        if inferencer is not None:
            assert isinstance(inferencer, DetectionInferencer)
        super().__init__(inferencer)
        self.jupyter_visualizer = None

    def _get_images_names_by_inference(
        self,
        images_data: List[ImageData],
        score_threshold: float,
        minimum_iou: float,
        batch_size: int
    ) -> List[str]:
        images_data_gen = BatchGeneratorImageData(images_data,
                                                  batch_size=batch_size,
                                                  use_not_caught_elements_as_last_batch=True)
        pred_images_data = self.inferencer.predict(images_data_gen, score_threshold)
        images_data_matchings = [
            ImageDataMatching(image_data, pred_image_data, minimum_iou)
            for image_data, pred_image_data in zip(images_data, pred_images_data)
        ]
        images_names = [
            f"{image_data_matching.true_image_data.image_path.name} "
            f"[TP: {image_data_matching.get_detection_TP()}, "
            f"FP: {image_data_matching.get_detection_FP()}, "
            f"FN: {image_data_matching.get_detection_FN()}]"
            for image_data_matching in images_data_matchings
        ]
        return images_names

    def visualize(self,
                  images_data: List[ImageData],
                  score_threshold: float = None,
                  show_TP_FP_FN_with_minimum_iou: float = None,
                  batch_size: int = 16):

        images_data = copy.deepcopy(images_data)

        if self.inferencer is not None and show_TP_FP_FN_with_minimum_iou is not None:
            images_names = self._get_images_names_by_inference(
                images_data=images_data,
                score_threshold=score_threshold,
                minimum_iou=show_TP_FP_FN_with_minimum_iou,
                batch_size=batch_size
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
                    self.pred_image_data = self.inferencer.predict([self.batch], score_threshold)[0]
                self.i = i

            if self.inferencer is None:
                display(Image.fromarray(visualize_image_data(
                    self.true_image_data,
                    use_labels=False,
                    score_type=None
                )))
            else:
                if show_TP_FP_FN_with_minimum_iou is not None:
                    display(Image.fromarray(visualize_image_data_matching_side_by_side(
                        image_data_matching=ImageDataMatching(self.true_image_data, self.pred_image_data,
                                                              show_TP_FP_FN_with_minimum_iou),
                        error_type='detection',
                        true_use_labels=True, pred_use_labels=True,
                        true_score_type=None, pred_score_type='detection',
                        true_filter_by_error_types=self.jupyter_visualizer.choices.value.split('+'),
                        pred_filter_by_error_types=self.jupyter_visualizer.choices2.value.split('+')
                    )))
                else:
                    display(Image.fromarray(visualize_images_data_side_by_side(
                        image_data1=self.true_image_data,
                        image_data2=self.pred_image_data,
                        use_labels1=False, use_labels2=False,
                        score_type1=None, score_type2='detection',
                    )))

        self.jupyter_visualizer = JupyterVisualizer(
            images=range(len(images_data_gen)),
            images_names=images_names,
            display_fn=display_fn,
            choices=(
                ['TP+FN', 'TP', 'FN']
                if self.inferencer is not None and show_TP_FP_FN_with_minimum_iou is not None else []
            ),
            choices_description='GT',
            choices2=(
                ['TP+FP', 'TP', 'FP']
                if self.inferencer is not None and show_TP_FP_FN_with_minimum_iou is not None else []
            ),
            choices2_description='Prediction',
        )
        self.jupyter_visualizer.visualize()
