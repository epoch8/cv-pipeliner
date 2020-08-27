from typing import Literal, List, Callable

import numpy as np
import imutils
import cv2

from object_detection.utils import label_map_util
from object_detection.utils.visualization_utils import \
    visualize_boxes_and_labels_on_image_array

from two_stage_pipeliner.core.data import BboxData, ImageData


def draw_label_image(
    image: np.ndarray,
    base_label_image: np.ndarray,
    bbox_data: BboxData,
    inplace: bool = False
) -> np.ndarray:

    if not inplace:
        image = image.copy()

    bbox_data_size = max(bbox_data.xmax - bbox_data.xmin, bbox_data.ymax - bbox_data.ymin)
    resize = int(bbox_data_size / 1.5) 
    height, width, _ = base_label_image.shape
    if height <= width:
        label_image = imutils.resize(base_label_image, width=resize)
    else:
        label_image = imutils.resize(base_label_image, height=resize)

    x_offset = bbox_data.xmin - 20
    y_offset = bbox_data.ymax - label_image.shape[0]

    y_min, y_max = y_offset, y_offset + label_image.shape[0]
    x_min, x_max = x_offset, x_offset + label_image.shape[1]

    # Ensure that label image is inside image boundaries
    if y_max > image.shape[0]:
        y_min -= y_max - image.shape[0]
        y_max = image.shape[0]

    if x_max > image.shape[1]:
        x_min -= x_max - image.shape[1]
        x_max = image.shape[1]

    if x_min < 0:
        x_max -= x_min
        x_min = 0

    if y_min < 0:
        y_max -= y_min
        y_min = 0

    alpha_label_image = label_image[:, :, 3] / 255.0
    alpha_image = 1.0 - alpha_label_image

    for channel in range(0, 3):
        image[y_min:y_max, x_min:x_max, channel] = (
            alpha_label_image * label_image[:, :, channel]
            + alpha_image * image[y_min:y_max, x_min:x_max, channel]
        )

    if not inplace:
        return image


def visualize_image_data(
    image_data: ImageData,
    use_labels: bool = False,
    known_labels: List[str] = None,
    score_type: Literal['detection', 'classification'] = None,
    filter_by_label: List[str] = None,
    draw_base_labels_with_given_label_to_base_label_image: Callable[[str], np.ndarray] = None
) -> np.ndarray:
    image = image_data.open_image().copy()
    bboxes_data = image_data.bboxes_data
    if filter_by_label is not None:
        bboxes_data = [
            bbox_data for bbox_data in bboxes_data
            if bbox_data.label in filter_by_label
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
            groundtruth_box_visualization_color='lime'
    )

    if draw_base_labels_with_given_label_to_base_label_image is not None:
        for bbox_data in image_data.bboxes_data:
            base_label_image = draw_base_labels_with_given_label_to_base_label_image(bbox_data.label)
            draw_label_image(
                image=image,
                base_label_image=base_label_image,
                bbox_data=bbox_data,
                inplace=True
            )

    return image


def visualize_images_data_side_by_side(
    image_data1: ImageData,
    image_data2: ImageData,
    use_labels1: bool = False,
    use_labels2: bool = False,
    score_type1: Literal['detection', 'classification'] = None,
    score_type2: Literal['detection', 'classification'] = None,
    filter_by_labels1: List[str] = None,
    filter_by_labels2: List[str] = None,
    draw_base_labels_with_given_label_to_base_label_image: Callable[[str], np.ndarray] = None
) -> np.ndarray:

    if use_labels1 and use_labels2:
        labels1 = [bbox_data.label for bbox_data in image_data1.bboxes_data]
        labels2 = [bbox_data.label for bbox_data in image_data2.bboxes_data]
        known_labels = list(set(labels1 + labels2))
    else:
        known_labels = None

    true_ann_image = visualize_image_data(
        image_data=image_data1,
        use_labels=use_labels1,
        known_labels=known_labels,
        score_type=score_type1,
        filter_by_label=filter_by_labels1,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image
    )
    pred_ann_image = visualize_image_data(
        image_data=image_data2,
        use_labels=use_labels2,
        known_labels=known_labels,
        score_type=score_type2,
        filter_by_label=filter_by_labels2,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image
    )

    image = cv2.hconcat([true_ann_image, pred_ann_image])

    return image
