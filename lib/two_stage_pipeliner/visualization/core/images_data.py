from pathlib import Path
from typing import Union, List

import numpy as np
import imageio
import cv2

from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils.visualization_utils import \
    visualize_boxes_and_labels_on_image_array

from two_stage_pipeliner.metrics_counters.core.detection import get_df_detector_matchings
from brickit_ml.utils.data import ImageData


def visualize_image_data(
    image_data: ImageData,
    use_labels: bool = True,
    score_type: Union[None, 'detection', 'classification'] = None
) -> Image.Image:
    image = imageio.imread(image_data.image_path, pilmode="RGB")

    bboxes_data = image_data.bboxes_data
    labels = [bbox_data.label for bbox_data in bboxes_data]
    bboxes = np.array([
        (bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax)
        for bbox_data in bboxes_data
    ])
    if score_type == 'detection':
        scores = np.array([bbox_data.detection_score for bbox_data in bboxes_data])
        skip_scores = False
    elif score_type == 'classification':
        scores = np.array([bbox_data.classification_score for bbox_data in bboxes_data])
        skip_scores = False
    else:
        scores = [1.] * len(bboxes_data)
        skip_scores = True

    if use_labels:
        labels = [
            label
            for label in labels
        ]
        categories = [{
            "id": i,
            "name": class_name
        } for i, class_name in enumerate(set(labels))]
        class_name_to_id = {
            category['name']: category['id']
            for category in categories
        }
        classes = np.array([class_name_to_id[class_name]
                            for class_name in labels])
    else:
        categories = [
            {
                "id": 1,
                "name": ""
            }
        ]
        classes = np.array([1] * len(bboxes))
    category_index = label_map_util.create_category_index(categories)
    image = Image.fromarray(
        visualize_boxes_and_labels_on_image_array(
            image,
            bboxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=None,
            skip_scores=skip_scores,
            min_score_thresh=0.,
            groundtruth_box_visualization_color='lime'
        )
    )
    return image


def visualize_images_data_side_by_side(
    image_data1: ImageData,
    image_data2: ImageData,
    use_labels1: bool = True,
    use_labels2: bool = False,
    score_type1: Union[None, 'detection', 'classification'] = None,
    score_type2: Union[None, 'detection', 'classification'] = 'detection',
    filepath: Union[str, Path] = None
) -> Image.Image:
    true_ann_image = visualize_image_data(image_data=image_data1,
                                          use_labels=use_labels1,
                                          score_type=score_type1)
    pred_ann_image = visualize_image_data(image_data=image_data2,
                                          use_labels=use_labels2,
                                          score_type=score_type2)

    image = Image.fromarray(
        cv2.hconcat([np.array(true_ann_image),
                     np.array(pred_ann_image)])
    )

    if filepath:
        image.save(filepath)
    else:
        return image
