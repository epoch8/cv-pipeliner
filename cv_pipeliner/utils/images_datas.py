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
            for bbox_data in rotated_image_data.bboxes_data
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
            for bbox_data in rotated_image_data.bboxes_data
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
        bbox_data.xmin = max(0, min(int(bbox_data.xmin * (new_width / old_width), new_width-1)))
        bbox_data.ymin = max(0, min(int(bbox_data.ymin * (new_height / old_height)), new_height-1))
        bbox_data.xmax = max(0, min(int(bbox_data.xmax * (new_width / old_width)), new_width-1))
        bbox_data.ymax = max(0, min(int(bbox_data.ymax * (new_height / old_height)), new_height-1))
        bbox_data.keypoints[:, 0] = (bbox_data.keypoints[:, 0] * (new_width / old_width)).astype(int)
        bbox_data.keypoints[:, 1] = (bbox_data.keypoints[:, 1] * (new_height / old_height)).astype(int)
        bbox_data.keypoints = bbox_data.keypoints.astype(int)
        keypoints = []
        for (x, y) in bbox_data.keypoints:
            x = max(0, min(x, new_width-1))
            y = max(0, min(y, new_height-1))
            keypoints.append([x, y])
        bbox_data.keypoints = np.array(keypoints).reshape(-1, 2)
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            resize_coords(additional_bbox_data)
    for bbox_data in image_data.bboxes_data:
        resize_coords(bbox_data)
    image_data.keypoints[:, 0] = (image_data.keypoints[:, 0] * (new_width / old_width)).astype(int)
    image_data.keypoints[:, 1] = (image_data.keypoints[:, 1] * (new_height / old_height)).astype(int)
    image_data.image_path = None
    image_data.image = image

    return image_data


def crop_image_data(
    image_data: ImageData,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    allow_negative_and_large_coords: bool,
    remove_bad_coords: bool,
) -> ImageData:

    assert 0 <= xmin and 0 <= ymin
    assert xmin <= xmax and ymin <= ymax

    image_data = copy.deepcopy(image_data)
    image = image_data.open_image()
    height, width, _ = image.shape

    assert xmax <= width and ymax <= height

    image = image[ymin:ymax, xmin:xmax]
    new_height, new_width, _ = image.shape

    def resize_coords(bbox_data: BboxData):
        bbox_data.xmin = bbox_data.xmin - xmin
        bbox_data.ymin = bbox_data.ymin - ymin
        bbox_data.xmax = bbox_data.xmax - xmin
        bbox_data.ymax = bbox_data.ymax - ymin
        bbox_data.keypoints[:, 0] -= xmin
        bbox_data.keypoints[:, 1] -= ymin
        if not allow_negative_and_large_coords:
            bbox_data.xmin = max(0, min(bbox_data.xmin, new_width-1))
            bbox_data.ymin = max(0, min(bbox_data.ymin, new_height-1))
            bbox_data.xmax = max(0, min(bbox_data.xmax, new_width-1))
            bbox_data.ymax = max(0, min(bbox_data.ymax, new_height-1))
            keypoints = []
            for (x, y) in bbox_data.keypoints:
                x = max(0, min(x, new_width-1))
                y = max(0, min(y, new_height-1))
                keypoints.append([x, y])
            bbox_data.keypoints = np.array(keypoints).reshape(-1, 2)
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            resize_coords(additional_bbox_data)
    for bbox_data in image_data.bboxes_data:
        resize_coords(bbox_data)

    keypoints = []
    for (x, y) in image_data.keypoints:
        x = max(0, min(x, new_width-1))
        y = max(0, min(y, new_height-1))
        keypoints.append([x, y])
    image_data.keypoints = np.array(keypoints).reshape(-1, 2)

    def if_bbox_data_inside_crop(bbox_data: BboxData):
        bbox_data.keypoints = bbox_data.keypoints[
            (
                (bbox_data.keypoints[:, 0] >= 0) &
                (bbox_data.keypoints[:, 1] >= 0) &
                (bbox_data.keypoints[:, 0] < new_height) &
                (bbox_data.keypoints[:, 1] < new_width)
            )
        ]
        bbox_data.additional_bboxes_data = [
            additional_bbox_data
            for additional_bbox_data in bbox_data.additional_bboxes_data
            if if_bbox_data_inside_crop(additional_bbox_data)
        ]
        return (
            bbox_data.xmin >= 0 and
            bbox_data.ymin >= 0 and
            bbox_data.xmax < new_width and
            bbox_data.ymax < new_height and
            bbox_data.xmin < bbox_data.xmax and
            bbox_data.ymin < bbox_data.ymax
        )

    if remove_bad_coords:
        image_data.bboxes_data = [
            bbox_data
            for bbox_data in image_data.bboxes_data
            if if_bbox_data_inside_crop(bbox_data)
        ]
        image_data.keypoints = image_data.keypoints[
            (
                (image_data.keypoints[:, 0] >= 0) &
                (image_data.keypoints[:, 1] >= 0) &
                (image_data.keypoints[:, 0] < new_height) &
                (image_data.keypoints[:, 1] < new_width)
            )
        ]

    image_data.image_path = None
    image_data.image = image

    return image_data


def apply_perspective_transform_to_points(
    points: List[Tuple[int, int]],
    perspective_matrix: np.ndarray,
    result_width: int,
    result_height: int,
    allow_negative_and_large_coords: bool,
    remove_bad_coords: bool
):
    points = np.array(points)
    if len(points) == 0:
        return points
    transformed_points = cv2.perspectiveTransform(
        points.reshape(1, -1, 2).astype(np.float32),
        perspective_matrix
    ).reshape(-1, 2).astype(int)
    if not allow_negative_and_large_coords:
        transformed_points_without_bad_coords = []
        for (x, y) in points:
            x = max(0, min(x, result_width-1))
            y = max(0, min(y, result_height-1))
            transformed_points_without_bad_coords.append([x, y])
        transformed_points = np.array(transformed_points_without_bad_coords)
    if remove_bad_coords:
        transformed_points = transformed_points[
            (transformed_points[:, 0] >= 0) & (transformed_points[:, 1] >= 0) &
            (transformed_points[:, 0] < result_width) & (transformed_points[:, 1] < result_height)
        ]
    return transformed_points


def apply_perspective_transform_to_bbox_data(
    bbox_data: BboxData,
    perspective_matrix: np.ndarray,
    result_width: int,
    result_height: int,
    allow_negative_and_large_coords: bool,
    remove_bad_coords: bool
) -> BboxData:
    bbox_points = np.array([
        [bbox_data.xmin, bbox_data.ymin],
        [bbox_data.xmin, bbox_data.ymax],
        [bbox_data.xmax, bbox_data.ymin],
        [bbox_data.xmax, bbox_data.ymax]
    ])
    transformed_points = cv2.perspectiveTransform(
        bbox_points.reshape(1, -1, 2).astype(np.float32),
        perspective_matrix
    ).reshape(-1, 2).astype(int)
    transformed_xmin = np.min(transformed_points[:, 0])
    transformed_ymin = np.min(transformed_points[:, 1])
    transformed_xmax = np.max(transformed_points[:, 0])
    transformed_ymax = np.max(transformed_points[:, 1])
    if not allow_negative_and_large_coords:
        transformed_xmin = max(0, min(transformed_xmin, result_width-1))
        transformed_ymin = max(0, min(transformed_ymin, result_height-1))
        transformed_xmax = max(0, min(transformed_xmax, result_width-1))
        transformed_ymax = max(0, min(transformed_ymax, result_height-1))
    if remove_bad_coords and not (
        transformed_xmin >= 0 and transformed_ymin >= 0 and
        transformed_xmax < result_width and transformed_ymax < result_height
        and transformed_xmin < transformed_xmax and transformed_ymin < transformed_ymax
    ):
        return None

    transformed_bbox_data = copy.deepcopy(bbox_data)
    transformed_bbox_data.xmin = transformed_xmin
    transformed_bbox_data.ymin = transformed_ymin
    transformed_bbox_data.xmax = transformed_xmax
    transformed_bbox_data.ymax = transformed_ymax
    transformed_bbox_data.keypoints = apply_perspective_transform_to_points(
        transformed_bbox_data.keypoints, perspective_matrix, result_height, result_height,
        allow_negative_and_large_coords, remove_bad_coords
    )
    transformed_bbox_data.additional_bboxes_data = [
        apply_perspective_transform_to_bbox_data(
            additional_bbox_data, perspective_matrix, result_height, result_height,
            allow_negative_and_large_coords, remove_bad_coords
        )
        for additional_bbox_data in transformed_bbox_data.additional_bboxes_data
    ]
    transformed_bbox_data.additional_bboxes_data = [
        additional_bbox_data
        for additional_bbox_data in transformed_bbox_data.additional_bboxes_data
        if additional_bbox_data is not None
    ]
    return transformed_bbox_data


def get_perspective_matrix_for_base_keypoints(
    base_keypoints: Tuple[
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int],
        Tuple[int, int]
    ]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    base_keypoints = np.array(base_keypoints, dtype=np.float32)
    (top_left, top_right, bottom_right, bottom_left) = base_keypoints
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    result_width = max(int(width_a), int(width_b))
    result_height = max(int(height_a), int(height_b))
    transformed_points = np.array([
        [0, 0],
        [result_width - 1, 0],
        [result_width - 1, result_height - 1],
        [0, result_height - 1]
    ], dtype=np.float32)
    perspective_matrix = cv2.getPerspectiveTransform(base_keypoints, transformed_points)
    return perspective_matrix, (result_width, result_height)


def apply_perspective_transform_to_image_data(
    image_data: ImageData,
    perspective_matrix: np.ndarray,
    result_width: int,
    result_height: int,
    allow_negative_and_large_coords: bool,
    remove_bad_coords: bool
) -> ImageData:
    image = image_data.open_image()
    image = cv2.warpPerspective(image, perspective_matrix, (result_width, result_height))

    image_data = copy.deepcopy(image_data)
    image_data.keypoints = apply_perspective_transform_to_points(
        image_data.keypoints, perspective_matrix, result_width, result_height,
        allow_negative_and_large_coords, remove_bad_coords
    )
    image_data.bboxes_data = [
        apply_perspective_transform_to_bbox_data(
            bbox_data, perspective_matrix, result_width, result_height,
            allow_negative_and_large_coords, remove_bad_coords
        )
        for bbox_data in image_data.bboxes_data
    ]
    image_data.bboxes_data = [
        bbox_data
        for bbox_data in image_data.bboxes_data
        if bbox_data is not None
    ]
    image_data.image_path = None
    image_data.image = image

    return image_data
