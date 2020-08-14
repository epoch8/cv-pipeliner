from pathlib import Path
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from two_stage_pipeliner.core.data import BboxData
from two_stage_pipeliner.utils.images import get_img_from_fig


def visualize_bboxes_data(
    bboxes_data: List[BboxData],
    class_name: str,
    visualize_size: int = None,
    pred_bboxes_data: List[BboxData] = None,
    type_only: Union['TP+FP', 'TP', 'FP'] = 'TP+FP',
    use_random: bool = False,
    filepath: Union[str, Path] = None,
) -> Image.Image:
    true_labels = np.array(
        [bbox_data.label for bbox_data in bboxes_data]
    )
    if pred_bboxes_data is not None:
        assert len(bboxes_data) == len(pred_bboxes_data)
        pred_labels = np.array(
            [bbox_data.label for bbox_data in pred_bboxes_data]
        )

    if class_name != 'all':
        data_by_class_name = (true_labels == class_name)
    else:
        data_by_class_name = np.array([True] * len(bboxes_data))

    if pred_bboxes_data is not None and type_only is not None:
        if type_only == 'TP':
            data_by_class_name = (
                data_by_class_name & (true_labels == pred_labels)
            )
        elif type_only == 'FP':
            data_by_class_name = (
                data_by_class_name & (true_labels != pred_labels)
            )

    data_by_class_name_idxs = np.where(data_by_class_name)[0]
    if visualize_size is None:
        visualize_size = len(data_by_class_name_idxs)

    size = min(visualize_size, len(data_by_class_name_idxs))

    if use_random:
        idxs = np.random.choice(data_by_class_name_idxs,
                                size=size,
                                replace=False)
    else:
        idxs = data_by_class_name_idxs[:size]

    figsize = 5
    if size >= 10:
        figsize += 20
    fig, axes = plt.subplots(int(np.ceil(size/5)), 5,
                             figsize=(20, figsize + size // 5))

    for idx, ax in zip(idxs, axes.flatten()):
        bbox_data = bboxes_data[idx]
        bbox = bbox_data.open_image_bbox()
        label = bbox_data.label
        bbox = np.array(bbox)
        ax.imshow(bbox)
        ax.axis('off')
        titles = []
        if label is not None:
            titles.append(f"true: {label}")
        if pred_bboxes_data is not None:
            pred_bbox_data = pred_bboxes_data[idx]
            pred_label = pred_bbox_data.label
            titles.append(f"pred: {pred_label}")
        title = '\n'.join(titles)
        ax.set_title(title)

    fig.suptitle(class_name, fontsize=20)

    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        image = get_img_from_fig(fig)
        plt.close()
        image = Image.fromarray(image)
        return image
