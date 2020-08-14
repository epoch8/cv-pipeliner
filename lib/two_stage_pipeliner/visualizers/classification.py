from typing import List
from collections import Counter

import numpy as np
from IPython.display import display

from two_stage_pipeliner.core.data import BboxData
from two_stage_pipeliner.core.visualizer import Visualizer
from two_stage_pipeliner.core.batch_generator import BatchGeneratorBboxData
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.utils.jupyter_visualizer import JupyterVisualizer
from two_stage_pipeliner.visualizers.core.bboxes_data import visualize_bboxes_data


class ClassificationVisualizer(Visualizer):
    def __init__(self, inferencer: DetectionInferencer = None):
        Visualizer.__init__(self, inferencer)
        self.jupyter_visualizer = None

    def visualize(self,
                  n_bboxes_data: List[List[BboxData]],
                  visualize_size: int = 50,
                  use_all_data: bool = False,
                  show_TP_FP: bool = False):
        if use_all_data:
            images_names = ['all']
            n_bboxes_data = [bbox_data for bboxes_data in n_bboxes_data for bbox_data in bboxes_data]
            bboxes_data_gen = BatchGeneratorBboxData([n_bboxes_data], batch_size=1)
        else:
            if self.inferencer is not None and show_TP_FP:
                bboxes_data_gen = BatchGeneratorBboxData(n_bboxes_data, batch_size=1)
                n_pred_bboxes_data = self.inferencer.predict(bboxes_data_gen)

                n_true_labels = [
                    np.array([bbox_data.label for bbox_data in bboxes_data])
                    for bboxes_data in n_bboxes_data
                ]
                n_pred_labels = [
                    np.array([bbox_data.label for bbox_data in bboxes_data])
                    for bboxes_data in n_pred_bboxes_data
                ]
                n_TP = [
                    np.sum(true_labels == pred_labels)
                    for true_labels, pred_labels in zip(n_true_labels, n_pred_labels)
                ]
                n_FP = [
                    np.sum(true_labels != pred_labels)
                    for true_labels, pred_labels in zip(n_true_labels, n_pred_labels)
                ]

                images_names = [
                    f"{bboxes_data[0].image_path.name} [TP: {TP}, FP: {FP}]"
                    for bboxes_data, TP, FP in zip(n_bboxes_data, n_TP, n_FP)
                ]
            else:
                images_names = [bboxes_data[0].image_path.name for bboxes_data in n_bboxes_data]
            bboxes_data_gen = BatchGeneratorBboxData(n_bboxes_data, batch_size=1)

        self.i = None

        def display_fn(i):
            if self.i is None or i != self.i:
                self.batch = bboxes_data_gen[i]
                self.true_bboxes_data = self.batch[0]
                if self.inferencer is not None:
                    self.pred_bboxes_data = self.inferencer.predict([self.batch])[0]
                else:
                    self.pred_bboxes_data = None

                true_labels = np.array([bbox_data.label for bbox_data in self.true_bboxes_data])
                if self.inferencer is not None and show_TP_FP:
                    pred_labels = np.array([bbox_data.label for bbox_data in self.pred_bboxes_data])
                    label_to_TP = {
                        label: np.sum((true_labels == pred_labels) & (true_labels == label))
                        for label in set(true_labels)
                    }
                    label_to_FP = {
                        label: np.sum((true_labels != pred_labels) & (true_labels == label))
                        for label in set(true_labels)
                    }
                    label_to_TP['all'] = np.sum(list(label_to_TP.values()))
                    label_to_FP['all'] = np.sum(list(label_to_FP.values()))
                label_to_count = dict(Counter(true_labels))
                label_to_count['all'] = len(true_labels)
                class_names = sorted(
                    list(set(['all'] + list(true_labels))),
                    key=lambda x: label_to_count[x],
                    reverse=True
                )
                if self.inferencer is not None and show_TP_FP:
                    class_names_visible = [
                        f"{class_name} [TP: {label_to_TP[class_name]}, FP:{label_to_FP[class_name]}]"
                        for class_name in class_names
                    ]
                else:
                    class_names_visible = [
                        f"{class_name} [{label_to_count[class_name]} total]"
                        for class_name in class_names
                    ]
                self.jupyter_visualizer.choices.change_options = True
                self.jupyter_visualizer.choices.options = list(zip(class_names_visible, class_names))
                self.jupyter_visualizer.choices.value = 'all'
                self.jupyter_visualizer.choices.change_options = False
                self.i = i

            if self.jupyter_visualizer.choices2 is not None:
                type_only = self.jupyter_visualizer.choices2.value
            else:
                type_only = 'TP+FP'

            display(visualize_bboxes_data(
                bboxes_data=self.true_bboxes_data,
                class_name=self.jupyter_visualizer.choices.value,
                visualize_size=visualize_size,
                pred_bboxes_data=self.pred_bboxes_data,
                type_only=type_only
            ))

        if self.jupyter_visualizer is not None:
            del self.jupyter_visualizer
        self.jupyter_visualizer = JupyterVisualizer(
            images=range(len(bboxes_data_gen)),
            images_names=images_names,
            choices=['all'],
            choices_description='class_name',
            choices2=['TP+FP', 'TP', 'FP'] if self.inferencer is not None and show_TP_FP else "",
            choices2_description='type',
            display_fn=display_fn
        )
        self.jupyter_visualizer.visualize()
