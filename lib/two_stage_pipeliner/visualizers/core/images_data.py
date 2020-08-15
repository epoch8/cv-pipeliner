from typing import Union, List

import numpy as np
import imageio
import cv2

from object_detection.utils import label_map_util
from object_detection.utils.visualization_utils import \
    visualize_boxes_and_labels_on_image_array

from brickit_ml.utils.data import ImageData


def visualize_image_data(
    image_data: ImageData,
    use_labels: bool = True,
    known_labels: List[str] = None,
    score_type: Union[None, 'detection', 'classification'] = None,
    filter_by: List[str] = None
) -> np.ndarray:
    if image_data.image is None:
        image = imageio.imread(image_data.image_path, pilmode="RGB")
    else:
        image = image_data.image

    bboxes_data = image_data.bboxes_data
    if filter_by is not None:
        bboxes_data = [
            bbox_data for bbox_data in bboxes_data
            if bbox_data.label in filter_by
        ]
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
        if known_labels is None:
            known_labels = labels
        categories = [{
            "id": i,
            "name": class_name
        } for i, class_name in enumerate(set(known_labels))]
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
    image = visualize_boxes_and_labels_on_image_array(
            image,
            bboxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=None,
            skip_scores=skip_scores,
            min_score_thresh=0.,
            groundtruth_box_visualization_color='lime',
    )
    return image


def visualize_images_data_side_by_side(
    image_data1: ImageData,
    image_data2: ImageData,
    use_labels1: bool = True,
    use_labels2: bool = False,
    score_type1: Union[None, 'detection', 'classification'] = None,
    score_type2: Union[None, 'detection', 'classification'] = 'detection',
    filter_by1: List[str] = None,
    filter_by2: List[str] = None
) -> np.ndarray:

    if use_labels1 and use_labels2:
        labels1 = [bbox_data.label for bbox_data in image_data1.bboxes_data]
        labels2 = [bbox_data.label for bbox_data in image_data2.bboxes_data]
        known_labels = list(set(labels1 + labels2))
    else:
        known_labels = None

    true_ann_image = visualize_image_data(image_data=image_data1,
                                          use_labels=use_labels1,
                                          known_labels=known_labels,
                                          score_type=score_type1,
                                          filter_by=filter_by1)
    pred_ann_image = visualize_image_data(image_data=image_data2,
                                          use_labels=use_labels2,
                                          known_labels=known_labels,
                                          score_type=score_type2,
                                          filter_by=filter_by2)

    image = cv2.hconcat([true_ann_image, pred_ann_image])

    return image
