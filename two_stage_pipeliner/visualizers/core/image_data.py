import collections
from typing import Literal, List, Callable, Tuple

import numpy as np
import imutils
import cv2
from PIL import Image

from object_detection.utils.visualization_utils import (
    STANDARD_COLORS, draw_bounding_box_on_image
)

from two_stage_pipeliner.core.data import BboxData, ImageData
from two_stage_pipeliner.utils.images_datas import get_image_data_filtered_by_labels


# Taken from object_detection.utils.visualization_utils
def visualize_boxes_and_labels_on_image_array(
    image: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    labels: List[str],
    scores: List[float],
    use_normalized_coordinates=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    known_labels: List[str] = None,
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
      image: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      labels: a numpy array of shape [N]. Note that class indices are 1-based.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      use_normalized_coordinates: whether boxes is to be interpreted as
        normalized coordinates or not.
      line_thickness: integer (default: 4) controlling line width of the boxes.
      groundtruth_box_visualization_color: box color for visualizing groundtruth
        boxes
      known_labels: a list of known labels. If given, bboxes colors will be chosen by this list.
      skip_boxes: whether to skip the drawing of bounding boxes.
      skip_scores: whether to skip score when drawing a single detection
      skip_labels: whether to skip label when drawing a single detection

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    scores = np.array(scores)

    if known_labels is not None:
        assert all(label in known_labels for label in labels)
    else:
        known_labels = sorted(list(set(labels)))

    label_to_id = {label: id_ for id_, label in enumerate(known_labels)}
    bbox_to_display_str = collections.defaultdict(list)
    bbox_to_color = collections.defaultdict(str)

    for i in range(len(bboxes)):
        bbox = tuple(bboxes[i].tolist())
        if skip_labels:
            bbox_to_color[bbox] = groundtruth_box_visualization_color
        else:
            display_str = ''
            if not skip_labels:
                display_str = str(labels[i])
            if not skip_scores:
                if not display_str:
                    display_str = f'{round(100*scores[i])}%'
                else:
                    display_str = f'{display_str}: {round(100*scores[i])}%'
            bbox_to_display_str[bbox].append(display_str)
            bbox_to_color[bbox] = STANDARD_COLORS[label_to_id[labels[i]] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    for bbox in bboxes:
        bbox = tuple(bbox.tolist())
        ymin, xmin, ymax, xmax = bbox
        draw_bounding_box_on_image(
            image=image_pil,
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax,
            color=bbox_to_color[bbox],
            thickness=line_thickness,
            display_str_list=bbox_to_display_str[bbox],
            use_normalized_coordinates=use_normalized_coordinates
        )
    image = np.array(image_pil)
    return image


def draw_label_image(
    image: np.ndarray,
    base_label_image: np.ndarray,
    bbox_data: BboxData,
    inplace: bool = False
) -> np.ndarray:

    if not inplace:
        image = image.copy()

    bbox_data_size = max(bbox_data.xmax - bbox_data.xmin, bbox_data.ymax - bbox_data.ymin)
    resize = min(int(bbox_data_size / 1.5), int(max(image.shape) / 20))

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
    score_type: Literal['detection', 'classification'] = None,
    filter_by_labels: List[str] = None,
    known_labels: List[str] = None,
    draw_base_labels_with_given_label_to_base_label_image: Callable[[str], np.ndarray] = None,
) -> np.ndarray:
    image_data = get_image_data_filtered_by_labels(
        image_data=image_data,
        filter_by_labels=filter_by_labels
    )
    image = image_data.open_image()
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
        scores = None
        skip_scores = True

    image = visualize_boxes_and_labels_on_image_array(
            image=image,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            use_normalized_coordinates=False,
            skip_scores=skip_scores,
            skip_labels=not use_labels,
            groundtruth_box_visualization_color='lime',
            known_labels=known_labels
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
    known_labels: List[str] = None,
    draw_base_labels_with_given_label_to_base_label_image: Callable[[str], np.ndarray] = None
) -> np.ndarray:

    true_ann_image = visualize_image_data(
        image_data=image_data1,
        use_labels=use_labels1,
        score_type=score_type1,
        filter_by_labels=filter_by_labels1,
        known_labels=known_labels,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
    )
    pred_ann_image = visualize_image_data(
        image_data=image_data2,
        use_labels=use_labels2,
        score_type=score_type2,
        filter_by_labels=filter_by_labels2,
        known_labels=known_labels,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
    )

    image = cv2.hconcat([true_ann_image, pred_ann_image])

    return image
