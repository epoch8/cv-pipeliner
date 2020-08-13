from typing import List
from IPython.display import display

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.core.visualizer import Visualizer
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.utils.jupyter_visualizer import JupyterVisualizer
from two_stage_pipeliner.visualization.core.images_data import visualize_image_data, \
    visualize_images_data_side_by_side


class DetectionVisualizer(Visualizer):
    def __init__(self, inferencer: DetectionInferencer = None):
        Visualizer.__init__(self, inferencer)
        self.jupyter_visualizer = None

    def visualize(self,
                  images_data: List[ImageData],
                  score_threshold: float = None):
        image_names = [image_data.image_path.name for image_data in images_data]
        images_data_gen = BatchGeneratorImageData(images_data, batch_size=1)

        def display_fn(batch):
            true_image_data = batch[0]
            if self.inferencer is None:
                display(visualize_image_data(
                    true_image_data,
                    use_labels=False,
                    score_type=None
                ))
            else:
                pred_image_data = self.inferencer.predict([batch], score_threshold)[0]
                display(visualize_images_data_side_by_side(
                    true_image_data, pred_image_data,
                    use_labels1=False, use_labels2=False,
                    score_type1=None, score_type2='detection'
                ))

        if self.jupyter_visualizer is not None:
            del self.jupyter_visualizer
        self.jupyter_visualizer = JupyterVisualizer(
            images=images_data_gen,
            images_names=image_names,
            display_fn=display_fn
        )
        self.jupyter_visualizer.visualize()
