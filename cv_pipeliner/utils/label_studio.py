from typing import Dict

import imagesize
from cv_pipeliner.core.data import BboxData, ImageData
import numpy as np


def parse_rectangle_labels_to_bbox_data(
    rectangle_label: Dict
) -> BboxData:
    original_height = rectangle_label['original_height']
    original_width = rectangle_label['original_width']
    height = rectangle_label['value']['height']
    width = rectangle_label['value']['width']
    xmin = rectangle_label['value']['x']
    ymin = rectangle_label['value']['y']
    label = rectangle_label['value']['rectanglelabels'][0]
    xmax = xmin + width
    ymax = ymin + height
    xmin = max(0, min(original_width - 1, xmin / 100 * original_width))
    ymin = max(0, min(original_height - 1, ymin / 100 * original_height))
    xmax = max(0, min(original_width - 1, xmax / 100 * original_width))
    ymax = max(0, min(original_height - 1, ymax / 100 * original_height))
    bbox_data = BboxData(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        label=label
    )
    return bbox_data


def convert_image_data_to_rectangle_labels(
    image_data: ImageData,
    from_name: str,
) -> Dict:
    if image_data.image_path is not None:
        im_width, im_height = imagesize.get(image_data.image_path)
    else:
        im_height, im_width, _ = image_data.open_image().shape
    rectangle_labels = []
    for bbox_data in image_data.bboxes_data:
        rectangle_labels.append({
            "original_width": im_width,
            "original_height": im_height,
            "image_rotation": 0,
            "value": {
                "x": bbox_data.xmin / im_width * 100,
                "y": bbox_data.ymin / im_height * 100,
                "width": (bbox_data.xmax - bbox_data.xmin) / im_width * 100,
                "height": (bbox_data.ymax - bbox_data.ymin) / im_height * 100,
                "rotation": 0,
                "rectanglelabels": [bbox_data.label]
            },
            "from_name": from_name,
            "to_name": "image",
            "type": "rectanglelabels"
        })
    return rectangle_labels


def parse_polygon_label_to_bbox_data(
    polygon_label: Dict
) -> BboxData:
    original_height = polygon_label['original_height']
    original_width = polygon_label['original_width']
    keypoints = []
    for (x, y) in polygon_label['value']['points']:
        x = x / 100 * polygon_label['original_width']
        y = y / 100 * polygon_label['original_height']
        keypoints.append([max(0, min(original_width - 1, x)), max(0, min(original_height - 1, y))])
    keypoints = np.array(keypoints)
    bbox_data = BboxData(
        xmin=np.min(keypoints[:, 0]),
        ymin=np.min(keypoints[:, 1]),
        xmax=np.max(keypoints[:, 0]),
        ymax=np.max(keypoints[:, 1]),
        keypoints=keypoints,
        label=polygon_label['value']['polygonlabels'][0]
    )
    return bbox_data


def convert_image_data_to_polygon_label(
    image_data: ImageData,
    from_name: str,
) -> Dict:
    if image_data.image_path is not None:
        im_width, im_height = imagesize.get(image_data.image_path)
    else:
        im_height, im_width, _ = image_data.open_image().shape
    rectangle_labels = [{
        "original_width": im_width,
        "original_height": im_height,
        "image_rotation": 0,
        "value": {
            "points": [
                [x * 100 / im_width, y * 100 / im_height]
                for x, y in image_data.keypoints
            ],
            "polygonlabels": [image_data.label]
        },
        "from_name": from_name,
        "to_name": "image",
        "type": "polygonlabels"
    }]
    return rectangle_labels
