import copy
from typing import List, Literal, Tuple

import numpy as np
import cv2

from cv_pipeliner.core.data import ImageData, BboxData
from PIL import Image


def get_image_data_filtered_by_labels(
    image_data: ImageData,
    filter_by_labels: List[str] = None,
    include: bool = True
) -> ImageData:
    if filter_by_labels is None or len(filter_by_labels) == 0:
        return image_data

    filter_by_labels = set(filter_by_labels)

    bboxes_data = [
        bbox_data for bbox_data in image_data.bboxes_data
        if (include and bbox_data.label in filter_by_labels) or (
            not include and bbox_data.label not in filter_by_labels
        )
    ]
    return ImageData(
        image_path=image_data.image_path,
        image=image_data.image,
        bboxes_data=bboxes_data
    )


def get_n_bboxes_data_filtered_by_labels(
    n_bboxes_data: List[List[BboxData]],
    filter_by_labels: List[str] = None,
    include: bool = True
) -> ImageData:
    if filter_by_labels is None or len(filter_by_labels) == 0:
        return n_bboxes_data

    filter_by_labels = set(filter_by_labels)

    n_bboxes_data = [
        [
            bbox_data for bbox_data in bboxes_data
            if (include and bbox_data.label in filter_by_labels) or (
                not include and bbox_data.label not in filter_by_labels
            )
        ]
        for bboxes_data in n_bboxes_data
    ]
    return n_bboxes_data


def rotate_keypoints(keypoints: Tuple[Tuple[int, int]], rotation_mat: np.ndarray):
    keypoints = np.array(keypoints)
    points = np.zeros((len(keypoints), 3))
    points[:, 0] = keypoints[:, 0]
    points[:, 1] = keypoints[:, 1]
    points[:, 2] = 1
    rotated_points = (rotation_mat @ points.T).astype(int).T
    return rotated_points[:, [0, 1]]


def rotate_keypoints90(
    keypoints: Tuple[Tuple[int, int]],
    factor: Literal[0, 1, 2, 3],
    width: int,
    height: int,
) -> np.ndarray:
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)"""
    rotated_keypoints = []
    for (x, y) in keypoints:
        if factor == 1:
            x, y = y, (width - 1) - x
        elif factor == 2:
            x, y = (width - 1) - x, (height - 1) - y
        elif factor == 3:
            x, y = (height - 1) - y, x
        rotated_keypoints.append([x, y])
    return np.array(rotated_keypoints)


def rotate_bbox_data(
    bbox_data: BboxData,
    rotation_mat: np.ndarray
) -> BboxData:
    bbox_points = np.array([
        [bbox_data.xmin, bbox_data.ymin],
        [bbox_data.xmin, bbox_data.ymax],
        [bbox_data.xmax, bbox_data.ymin],
        [bbox_data.xmax, bbox_data.ymax]
    ])
    rotated_points = rotate_keypoints(bbox_points, rotation_mat)
    rotated_xmin = np.min(rotated_points[:, 0])
    rotated_ymin = np.min(rotated_points[:, 1])
    rotated_xmax = np.max(rotated_points[:, 0])
    rotated_ymax = np.max(rotated_points[:, 1])
    rotated_bbox_data = copy.deepcopy(bbox_data)
    rotated_bbox_data.xmin = rotated_xmin
    rotated_bbox_data.ymin = rotated_ymin
    rotated_bbox_data.xmax = rotated_xmax
    rotated_bbox_data.ymax = rotated_ymax
    rotated_bbox_data.keypoints = rotate_keypoints(rotated_bbox_data.keypoints, rotation_mat)
    rotated_bbox_data.additional_bboxes_data = [
        rotate_bbox_data(additional_bbox_data, rotation_mat)
        for additional_bbox_data in rotated_bbox_data.additional_bboxes_data
    ]

    return rotated_bbox_data


def rotate_bbox_data90(
    bbox_data: BboxData,
    factor: Literal[0, 1, 2, 3],
    width: int,
    height: int,
) -> BboxData:
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)"""
    rotated_bbox_data = copy.deepcopy(bbox_data)
    rotated_bbox_data.keypoints = rotate_keypoints90(bbox_data.keypoints, factor, width, height)
    xmin, ymin, xmax, ymax = rotated_bbox_data.coords
    if factor == 1:
        xmin, ymin, xmax, ymax = ymin, width - xmax, ymax, width - xmin
    elif factor == 2:
        xmin, ymin, xmax, ymax = width - xmax, height - ymax, width - xmin, height - ymin
    elif factor == 3:
        xmin, ymin, xmax, ymax = height - ymax, xmin, height - ymin, xmax
    rotated_bbox_data.xmin = xmin
    rotated_bbox_data.ymin = ymin
    rotated_bbox_data.xmax = xmax
    rotated_bbox_data.ymax = ymax
    rotated_bbox_data.additional_bboxes_data = [
        rotate_bbox_data90(additional_bbox_data, factor, width, height)
        for additional_bbox_data in rotated_bbox_data.additional_bboxes_data
    ]
    return rotated_bbox_data


def rotate_image_data(
    image_data: ImageData,
    angle: float
):
    if abs(angle) <= 1e-6:
        return image_data

    image = image_data.open_image()
    height, width, _ = image.shape
    image_center = width // 2, height // 2

    angle_to_factor = {
        0: 0,
        90: 1,
        180: 2,
        270: 3
    }
    angle = angle % 360
    rotated_image_data = copy.deepcopy(image_data)

    if angle in angle_to_factor:
        factor = angle_to_factor[angle]
        rotated_image = np.rot90(image, factor)
        rotated_image_data.keypoints = rotate_keypoints90(
            image_data.keypoints, factor, width, height
        )
        rotated_image_data.bboxes_data = [
            rotate_bbox_data90(bbox_data, factor, width, height)
            for bbox_data in image_data.bboxes_data
        ]
    else:
        # grab the rotation matrix
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        # compute the new bounding dimensions of the image
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        # adjust the rotation matrix to take into account translation
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
        rotated_image_data = copy.deepcopy(image_data)
        rotated_image_data.keypoints = rotate_keypoints(image_data.keypoints, rotation_mat)
        rotated_image_data.bboxes_data = [
            rotate_bbox_data(bbox_data, rotation_mat)
            for bbox_data in image_data.bboxes_data
        ]

    rotated_image_data.image_path = None  # It applies to all bboxes_data inside
    rotated_image_data.image = rotated_image

    return rotated_image_data


def thumbnail_image_data(
    image_data: ImageData,
    size: Tuple[int, int]
) -> ImageData:
    image_data = copy.deepcopy(image_data)
    image = image_data.open_image()
    old_height, old_width, _ = image.shape
    image = Image.fromarray(image)
    image.thumbnail(size)
    image = np.array(image)
    new_height, new_width, _ = image.shape

    def resize_coords(bbox_data: BboxData):
        bbox_data.xmin = int(bbox_data.xmin * (new_width / old_width))
        bbox_data.ymin = int(bbox_data.ymin * (new_height / old_height))
        bbox_data.xmax = int(bbox_data.xmax * (new_width / old_width))
        bbox_data.ymax = int(bbox_data.ymax * (new_height / old_height))
        bbox_data.keypoints[:, 0] *= (new_width / old_width)
        bbox_data.keypoints[:, 1] *= (new_height / old_height)
        bbox_data.keypoints = bbox_data.keypoints.astype(int)
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            resize_coords(additional_bbox_data)
    for bbox_data in image_data.bboxes_data:
        resize_coords(bbox_data)
    image_data.keypoints[:, 0] *= (new_width / old_width)
    image_data.keypoints[:, 1] *= (new_height / old_height)
    image_data.image_path = None
    image_data.image = image

    return image_data


def crop_image_data(
    image_data: ImageData,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int
) -> ImageData:

    assert 0 <= xmin and 0 <= ymin
    assert xmin <= xmax and ymin <= ymax

    image_data = copy.deepcopy(image_data)
    image = image_data.open_image()
    height, width, _ = image.shape

    assert xmax <= width and ymax <= height

    def if_bbox_data_inside_crop(bbox_data: BboxData):
        bbox_data.keypoints = bbox_data.keypoints[
            ~(
                bbox_data.keypoints[:, 0] >= xmax |
                bbox_data.keypoints[:, 1] >= ymax |
                bbox_data.keypoints[:, 0] <= xmin |
                bbox_data.keypoints[:, 1] <= ymin
            )
        ]
        bbox_data.additional_bboxes_data = [
            additional_bbox_data
            for additional_bbox_data in bbox_data.additional_bboxes_data
            if if_bbox_data_inside_crop(additional_bbox_data)
        ]
        return not (
            bbox_data.xmin >= xmax or
            bbox_data.ymin >= ymax or
            bbox_data.xmax <= xmin or
            bbox_data.ymax <= ymin
        )

    image_data.bboxes_data = [
        bbox_data
        for bbox_data in image_data.bboxes_data
        if if_bbox_data_inside_crop(bbox_data)
    ]
    image_data.keypoints = image_data.keypoints[
        ~(
            image_data.keypoints[:, 0] >= xmax |
            image_data.keypoints[:, 1] >= ymax |
            image_data.keypoints[:, 0] <= xmin |
            image_data.keypoints[:, 1] <= ymin
        )
    ]

    image = image[ymin:ymax, xmin:xmax]

    def resize_coords(bbox_data: BboxData):
        bbox_data.xmin = bbox_data.xmin - xmin
        bbox_data.ymin = bbox_data.ymin - ymin
        bbox_data.xmax = bbox_data.xmax - xmin
        bbox_data.ymax = bbox_data.ymax - ymin
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            resize_coords(additional_bbox_data)
    for bbox_data in image_data.bboxes_data:
        resize_coords(bbox_data)

    image_data.image_path = None
    image_data.image = image

    return image_data
