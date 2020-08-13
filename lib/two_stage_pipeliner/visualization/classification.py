from typing import List
from collections import Counter

from IPython.display import display

from two_stage_pipeliner.core.data import BboxData
from two_stage_pipeliner.core.visualizer import Visualizer
from two_stage_pipeliner.core.batch_generator import BatchGeneratorBboxData
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.utils.jupyter_visualizer import JupyterVisualizer
from two_stage_pipeliner.visualization.core.bboxes_data import visualize_bboxes_data


class ClassificationVisualizer(Visualizer):
    def __init__(self, inferencer: DetectionInferencer = None):
        Visualizer.__init__(self, inferencer)
        self.jupyter_visualizer = None

    def visualize(self,
                  n_bboxes_data: List[List[BboxData]],
                  visualize_size: int = 50,
                  use_all_data: bool = False,
                  errors_only: bool = False):
        if use_all_data:
            images_names = ['all']
            n_bboxes_data = [bbox_data for bboxes_data in n_bboxes_data for bbox_data in bboxes_data]
            bboxes_data_gen = BatchGeneratorBboxData([n_bboxes_data], batch_size=1)
        else:
            images_names = [bboxes_data[0].image_path.name for bboxes_data in n_bboxes_data]
            bboxes_data_gen = BatchGeneratorBboxData(n_bboxes_data, batch_size=1)
        self.i = None

        def display_fn(i):
            if self.i is None or i != self.i:
                self.batch = bboxes_data_gen[i]
                self.true_bboxes_data = self.batch[0]
                if self.inferencer is None:
                    self.pred_bboxes_data = None
                else:
                    self.pred_bboxes_data = self.inferencer.predict([self.batch])[0]

                true_labels = [bbox_data.label for bbox_data in self.true_bboxes_data]
                label_to_count = dict(Counter(true_labels))
                label_to_count['random'] = len(true_labels)

                class_names = sorted(
                    list(set(['random'] + true_labels)),
                    key=lambda x: label_to_count[x],
                    reverse=True
                )
                class_names_visible = [
                    f"{class_name} [{label_to_count[class_name]}]" for class_name in class_names
                ]
                self.jupyter_visualizer.choices.change_options = True
                self.jupyter_visualizer.choices.options = list(zip(class_names_visible, class_names))
                self.jupyter_visualizer.choices.value = 'random'
                self.jupyter_visualizer.choices.change_options = False
                self.i = i

            display(visualize_bboxes_data(
                bboxes_data=self.true_bboxes_data,
                class_name=self.jupyter_visualizer.choices.value,
                visualize_size=visualize_size,
                pred_bboxes_data=self.pred_bboxes_data,
                errors_only=errors_only
            ))

        if self.jupyter_visualizer is not None:
            del self.jupyter_visualizer
        self.jupyter_visualizer = JupyterVisualizer(
            images=range(len(bboxes_data_gen)),
            images_names=images_names,
            display_fn=display_fn
        )
        self.jupyter_visualizer.visualize()
