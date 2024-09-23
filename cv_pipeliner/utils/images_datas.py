import copy
from typing import List, Literal, Optional, Tuple, Union

import cv2
import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.utils.images import concat_images, get_thumbnail_resize


def get_image_data_filtered_by_labels(
    image_data: ImageData, filter_by_labels: List[str] = None, include: bool = True
) -> ImageData:
    if filter_by_labels is None or len(filter_by_labels) == 0:
        return image_data
    image_data = copy.deepcopy(image_data)

    filter_by_labels = set(filter_by_labels)

    image_data.bboxes_data = [
        bbox_data
        for bbox_data in image_data.bboxes_data
        if (include and bbox_data.label in filter_by_labels)
        or (not include and bbox_data.label not in filter_by_labels)
    ]
    return image_data


def get_n_bboxes_data_filtered_by_labels(
    n_bboxes_data: List[List[BboxData]], filter_by_labels: List[str] = None, include: bool = True
) -> ImageData:
    if filter_by_labels is None or len(filter_by_labels) == 0:
        return n_bboxes_data

    filter_by_labels = set(filter_by_labels)

    n_bboxes_data = [
        [
            bbox_data
            for bbox_data in bboxes_data
            if (include and bbox_data.label in filter_by_labels)
            or (not include and bbox_data.label not in filter_by_labels)
        ]
        for bboxes_data in n_bboxes_data
    ]
    return n_bboxes_data


def rotate_keypoints(keypoints: Tuple[Tuple[int, int]], rotation_mat: np.ndarray, new_width: int, new_height: int):
    keypoints = np.array(keypoints)
    points = np.zeros((len(keypoints), 3))
    points[:, 0] = keypoints[:, 0]
    points[:, 1] = keypoints[:, 1]
    points[:, 2] = 1
    rotated_points = (rotation_mat @ points.T).astype(int).T
    rotated_points = np.array(rotated_points).reshape(-1, 2)
    rotated_points[:, 0] = np.clip(rotated_points[:, 0], 0, new_width - 1)
    rotated_points[:, 1] = np.clip(rotated_points[:, 1], 0, new_height - 1)
    return rotated_points


def rotate_keypoints90(
    keypoints: Tuple[Tuple[int, int]],
    factor: Literal[0, 1, 2, 3],
    width: int,
    height: int,
) -> np.ndarray:
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)"""
    rotated_keypoints = []
    for x, y in keypoints:
        if factor == 1:
            x, y = y, (width - 1) - x
        elif factor == 2:
            x, y = (width - 1) - x, (height - 1) - y
        elif factor == 3:
            x, y = (height - 1) - y, x
        rotated_keypoints.append([x, y])
    return np.array(rotated_keypoints).reshape(-1, 2)


def _rotate_bbox_data(bbox_data: BboxData, rotation_mat: np.ndarray, new_width: int, new_height: int) -> BboxData:
    bbox_points = np.array(
        [
            [bbox_data.xmin, bbox_data.ymin],
            [bbox_data.xmin, bbox_data.ymax],
            [bbox_data.xmax, bbox_data.ymin],
            [bbox_data.xmax, bbox_data.ymax],
        ]
    )
    rotated_points = rotate_keypoints(bbox_points, rotation_mat, new_width, new_height)
    rotated_xmin = max(0, min(np.min(rotated_points[:, 0]), new_width - 1))
    rotated_ymin = max(0, min(np.min(rotated_points[:, 1]), new_height - 1))
    rotated_xmax = max(0, min(np.max(rotated_points[:, 0]), new_width - 1))
    rotated_ymax = max(0, min(np.max(rotated_points[:, 1]), new_height - 1))
    rotated_bbox_data = copy.deepcopy(bbox_data)
    rotated_bbox_data.xmin = rotated_xmin
    rotated_bbox_data.ymin = rotated_ymin
    rotated_bbox_data.xmax = rotated_xmax
    rotated_bbox_data.ymax = rotated_ymax
    rotated_bbox_data.keypoints = rotate_keypoints(rotated_bbox_data.keypoints, rotation_mat, new_width, new_height)
    if isinstance(rotated_bbox_data.mask, list):
        rotated_bbox_data.mask = [
            rotate_keypoints(polygon, rotation_mat, new_width, new_height) for polygon in rotated_bbox_data.mask
        ]
        for polygon in rotated_bbox_data.mask:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, new_width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, new_height - 1)
    keypoints = []
    for x, y in rotated_bbox_data.keypoints:
        x = max(0, min(x, new_width - 1))
        y = max(0, min(y, new_height - 1))
        keypoints.append([x, y])
    rotated_bbox_data.keypoints = np.array(keypoints).reshape(-1, 2)
    rotated_bbox_data.additional_bboxes_data = [
        _rotate_bbox_data(additional_bbox_data, rotation_mat, new_width, new_height)
        for additional_bbox_data in rotated_bbox_data.additional_bboxes_data
    ]
    rotated_bbox_data.cropped_image = None
    return rotated_bbox_data


def _rotate_bbox_data90(
    bbox_data: BboxData,
    factor: Literal[0, 1, 2, 3],
    width: int,
    height: int,
) -> BboxData:
    """Rotates a bounding box by 90 degrees CCW (see np.rot90)"""
    rotated_bbox_data = copy.deepcopy(bbox_data)
    rotated_bbox_data.keypoints = rotate_keypoints90(bbox_data.keypoints, factor, width, height)
    if isinstance(rotated_bbox_data.mask, list):
        rotated_bbox_data.mask = [
            rotate_keypoints90(polygon, factor, width, height) for polygon in rotated_bbox_data.mask
        ]
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
        _rotate_bbox_data90(additional_bbox_data, factor, width, height)
        for additional_bbox_data in rotated_bbox_data.additional_bboxes_data
    ]
    rotated_bbox_data.cropped_image = None
    return rotated_bbox_data


def rotate_image_data(
    image_data: ImageData,
    angle: float,
    warp_flags: Optional[int] = None,
    border_mode: Optional[int] = None,
    border_value: Tuple[int, int, int] = None,
    open_image: bool = True,
    exif_transpose: bool = False,
):
    rotated_image_data = copy.deepcopy(image_data)
    if abs(angle) <= 1e-6:
        if open_image:
            rotated_image_data.open_image(inplace=True, exif_transpose=exif_transpose)
        return rotated_image_data

    width, height = image_data.get_image_size()
    image = image_data.open_image(returns_none_if_empty=True, exif_transpose=exif_transpose) if open_image else None
    image_center = width // 2, height // 2

    angle_to_factor = {0: 0, 90: 1, 180: 2, 270: 3}
    angle = angle % 360

    if angle in angle_to_factor:
        factor = angle_to_factor[angle]
        rotated_image = np.rot90(image, factor) if image is not None else None
        rotated_image_data.keypoints = rotate_keypoints90(image_data.keypoints, factor, width, height)
        if isinstance(rotated_image_data.mask, list):
            rotated_image_data.keypoints = [
                rotate_keypoints90(polygon, factor, width, height) for polygon in rotated_image_data.mask
            ]
        rotated_image_data.bboxes_data = [
            _rotate_bbox_data90(bbox_data, factor, width, height) for bbox_data in rotated_image_data.bboxes_data
        ]
        new_width, new_height = (width, height) if (factor % 2 == 0) else (height, width)
    else:
        # grab the rotation matrix
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        # compute the new bounding dimensions of the image
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        # adjust the rotation matrix to take into account translation
        rotation_mat[0, 2] += new_width / 2 - image_center[0]
        rotation_mat[1, 2] += new_height / 2 - image_center[1]

        rotated_image = (
            cv2.warpAffine(
                image,
                rotation_mat,
                (new_width, new_height),
                flags=warp_flags,
                borderMode=border_mode,
                borderValue=border_value,
            )
            if image is not None
            else None
        )
        rotated_image_data = copy.deepcopy(image_data)
        rotated_image_data.keypoints = rotate_keypoints(image_data.keypoints, rotation_mat, new_height, new_width)
        if isinstance(rotated_image_data.mask, list):
            rotated_image_data.keypoints = [
                rotate_keypoints(polygon, rotation_mat, new_height, new_width) for polygon in rotated_image_data.mask
            ]
        keypoints = []
        for x, y in rotated_image_data.keypoints:
            x = max(0, min(x, new_width - 1))
            y = max(0, min(y, new_height - 1))
            keypoints.append([x, y])
        rotated_image_data.keypoints = np.array(keypoints).reshape(-1, 2)
        if isinstance(rotated_image_data.mask, list):
            for polygon in rotated_image_data.mask:
                polygon[:, 0] = np.clip(polygon[:, 0], 0, new_width - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, new_height - 1)
        rotated_image_data.bboxes_data = [
            _rotate_bbox_data(bbox_data, rotation_mat, new_height, new_width)
            for bbox_data in rotated_image_data.bboxes_data
        ]

    rotated_image_data.image_path = None  # It applies to all bboxes_data inside
    rotated_image_data.meta_height = new_height
    rotated_image_data.meta_width = new_width
    rotated_image_data.image = rotated_image

    return rotated_image_data


def resize_image_data(
    image_data: ImageData,
    size: Tuple[int, int],
    interpolation: Optional[int] = cv2.INTER_LINEAR,
    open_image: bool = True,
    exif_transpose: bool = False,
) -> ImageData:
    image_data = copy.deepcopy(image_data)
    old_width, old_height = image_data.get_image_size()
    new_width, new_height = size

    image = image_data.open_image(returns_none_if_empty=True, exif_transpose=exif_transpose) if open_image else None
    image = cv2.resize(image, size, interpolation=interpolation) if image is not None else None

    def resize_coords(bbox_data: BboxData):
        bbox_data.xmin = max(0, min(int(bbox_data.xmin * (new_width / old_width)), new_width - 1))
        bbox_data.ymin = max(0, min(int(bbox_data.ymin * (new_height / old_height)), new_height - 1))
        bbox_data.xmax = max(0, min(int(bbox_data.xmax * (new_width / old_width)), new_width - 1))
        bbox_data.ymax = max(0, min(int(bbox_data.ymax * (new_height / old_height)), new_height - 1))
        bbox_data.keypoints[:, 0] = (bbox_data.keypoints[:, 0] * (new_width / old_width)).astype(int)
        bbox_data.keypoints[:, 1] = (bbox_data.keypoints[:, 1] * (new_height / old_height)).astype(int)
        if isinstance(bbox_data.mask, list):
            for polygon in bbox_data.mask:
                polygon[:, 0] = (polygon[:, 0] * (new_width / old_width)).astype(int)
                polygon[:, 1] = (polygon[:, 1] * (new_height / old_height)).astype(int)
                polygon[:, 0] = np.clip(polygon[:, 0], 0, new_width - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, new_height - 1)
        bbox_data.keypoints = bbox_data.keypoints.astype(int)
        bbox_data.cropped_image = None
        keypoints = []
        for x, y in bbox_data.keypoints:
            x = max(0, min(x, new_width - 1))
            y = max(0, min(y, new_height - 1))
            keypoints.append([x, y])
        bbox_data.keypoints = np.array(keypoints).reshape(-1, 2)
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            resize_coords(additional_bbox_data)

    for bbox_data in image_data.bboxes_data:
        resize_coords(bbox_data)
    image_data.keypoints[:, 0] = (image_data.keypoints[:, 0] * (new_width / old_width)).astype(int)
    image_data.keypoints[:, 1] = (image_data.keypoints[:, 1] * (new_height / old_height)).astype(int)
    keypoints = []
    for x, y in image_data.keypoints:
        x = max(0, min(x, new_width - 1))
        y = max(0, min(y, new_height - 1))
        keypoints.append([x, y])
    if isinstance(image_data.mask, list):
        for polygon in image_data.mask:
            polygon[:, 0] = (polygon[:, 0] * (new_width / old_width)).astype(int)
            polygon[:, 1] = (polygon[:, 1] * (new_height / old_height)).astype(int)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, new_width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, new_height - 1)
    image_data.keypoints = np.array(keypoints).reshape(-1, 2)
    image_data.image_path = None
    image_data.image = image

    return image_data


def thumbnail_image_data(
    image_data: ImageData,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    interpolation: Optional[int] = cv2.INTER_LINEAR,
) -> ImageData:
    if isinstance(size, int):
        size = (size, size)
    new_width, new_height = get_thumbnail_resize(image_data.get_image_size(), size)
    return resize_image_data(image_data, (new_width, new_height), interpolation=interpolation)


def crop_image_data(
    image_data: ImageData,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    allow_negative_and_large_coords: bool,
    remove_bad_coords: bool,
    open_image: bool = True,
    exif_transpose: bool = False,
) -> ImageData:
    assert 0 <= xmin and 0 <= ymin
    assert xmin <= xmax and ymin <= ymax

    image_data = copy.deepcopy(image_data)
    image = image_data.open_image(returns_none_if_empty=True, exif_transpose=exif_transpose) if open_image else None
    width, height = image_data.get_image_size()
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(xmax, width - 1)
    ymax = min(ymax, height - 1)

    assert (
        xmin >= 0 and ymin >= 0 and xmin < xmax and ymin < ymax and xmax <= width - 1 and ymax <= height - 1
    ), f"Wrong arguments: {(xmin, ymin, xmax, ymax)=} ({width=}, {height=})"

    image = image[ymin:ymax, xmin:xmax] if image is not None else None
    new_width, new_height = max(0, min(width - 1, xmax - xmin)), max(0, min(height - 1, ymax - ymin))

    def resize_coords(bbox_data: BboxData):
        bbox_data.xmin = bbox_data.xmin - xmin
        bbox_data.ymin = bbox_data.ymin - ymin
        bbox_data.xmax = bbox_data.xmax - xmin
        bbox_data.ymax = bbox_data.ymax - ymin
        bbox_data.keypoints[:, 0] -= xmin
        bbox_data.keypoints[:, 1] -= ymin
        if isinstance(bbox_data.mask, list):
            for polygon in bbox_data.mask:
                polygon[:, 0] -= xmin
                polygon[:, 1] -= ymin
        bbox_data.cropped_image = None
        if not allow_negative_and_large_coords:
            bbox_data.xmin = max(0, min(bbox_data.xmin, new_width - 1))
            bbox_data.ymin = max(0, min(bbox_data.ymin, new_height - 1))
            bbox_data.xmax = max(0, min(bbox_data.xmax, new_width - 1))
            bbox_data.ymax = max(0, min(bbox_data.ymax, new_height - 1))
            keypoints = []
            for x, y in bbox_data.keypoints:
                x = max(0, min(x, new_width - 1))
                y = max(0, min(y, new_height - 1))
                keypoints.append([x, y])
            bbox_data.keypoints = np.array(keypoints).reshape(-1, 2)
            if isinstance(bbox_data.mask, list):
                for polygon in bbox_data.mask:
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, new_width - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, new_height - 1)
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            resize_coords(additional_bbox_data)

    for bbox_data in image_data.bboxes_data:
        resize_coords(bbox_data)

    keypoints = []
    for x, y in image_data.keypoints:
        x = max(0, min(x - xmin, new_width - 1))
        y = max(0, min(y - ymin, new_height - 1))
        keypoints.append([x, y])
    image_data.keypoints = np.array(keypoints).reshape(-1, 2)
    if isinstance(image_data.mask, list):
        for polygon in image_data.mask:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, new_width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, new_height - 1)

    def if_bbox_data_inside_crop(bbox_data: BboxData):
        bbox_data.keypoints = bbox_data.keypoints[
            (
                (bbox_data.keypoints[:, 0] >= 0)
                & (bbox_data.keypoints[:, 1] >= 0)
                & (bbox_data.keypoints[:, 0] < new_height)
                & (bbox_data.keypoints[:, 1] < new_width)
            )
        ]
        if isinstance(bbox_data.mask, list):
            bbox_data.mask = [
                polygon[
                    (
                        (polygon[:, 0] >= 0)
                        & (polygon[:, 1] >= 0)
                        & (polygon[:, 0] < new_height)
                        & (polygon[:, 1] < new_width)
                    )
                ]
                for polygon in bbox_data.mask
            ]
            bbox_data.mask = [polygon for polygon in bbox_data.mask if len(polygon) > 0]
        bbox_data.additional_bboxes_data = [
            additional_bbox_data
            for additional_bbox_data in bbox_data.additional_bboxes_data
            if if_bbox_data_inside_crop(additional_bbox_data)
        ]
        return (
            bbox_data.xmin >= 0
            and bbox_data.ymin >= 0
            and bbox_data.xmax < new_width
            and bbox_data.ymax < new_height
            and bbox_data.xmin < bbox_data.xmax
            and bbox_data.ymin < bbox_data.ymax
        )

    if remove_bad_coords:
        image_data.bboxes_data = [
            bbox_data for bbox_data in image_data.bboxes_data if if_bbox_data_inside_crop(bbox_data)
        ]
        image_data.keypoints = image_data.keypoints[
            (
                (image_data.keypoints[:, 0] >= 0)
                & (image_data.keypoints[:, 1] >= 0)
                & (image_data.keypoints[:, 0] < new_height)
                & (image_data.keypoints[:, 1] < new_width)
            )
        ]
        if isinstance(image_data.mask, list):
            image_data.mask = [
                polygon[
                    (
                        (polygon[:, 0] >= 0)
                        & (polygon[:, 1] >= 0)
                        & (polygon[:, 0] < new_height)
                        & (polygon[:, 1] < new_width)
                    )
                ]
                for polygon in image_data.mask
            ]
            image_data.mask = [polygon for polygon in image_data.mask if len(polygon) > 0]

    image_data.image_path = None
    image_data.image = image

    return image_data


def apply_perspective_transform_to_points(
    points: List[Tuple[int, int]],
    perspective_matrix: np.ndarray,
    result_width: int,
    result_height: int,
    allow_negative_and_large_coords: bool,
    remove_bad_coords: bool,
):
    points = np.array(points)
    if len(points) == 0:
        return points
    transformed_points = (
        cv2.perspectiveTransform(points.reshape(1, -1, 2).astype(np.float32), perspective_matrix)
        .reshape(-1, 2)
        .astype(int)
    )
    if not allow_negative_and_large_coords:
        transformed_points_without_bad_coords = []
        for x, y in points:
            x = max(0, min(x, result_width - 1))
            y = max(0, min(y, result_height - 1))
            transformed_points_without_bad_coords.append([x, y])
        transformed_points = np.array(transformed_points_without_bad_coords)
    if remove_bad_coords:
        transformed_points = transformed_points[
            (transformed_points[:, 0] >= 0)
            & (transformed_points[:, 1] >= 0)
            & (transformed_points[:, 0] < result_width)
            & (transformed_points[:, 1] < result_height)
        ]
    return transformed_points


def _apply_perspective_transform_to_bbox_data(
    bbox_data: BboxData,
    perspective_matrix: np.ndarray,
    result_width: int,
    result_height: int,
    allow_negative_and_large_coords: bool,
    remove_bad_coords: bool,
) -> BboxData:
    bbox_points = np.array(
        [
            [bbox_data.xmin, bbox_data.ymin],
            [bbox_data.xmin, bbox_data.ymax],
            [bbox_data.xmax, bbox_data.ymin],
            [bbox_data.xmax, bbox_data.ymax],
        ]
    )
    transformed_points = (
        cv2.perspectiveTransform(bbox_points.reshape(1, -1, 2).astype(np.float32), perspective_matrix)
        .reshape(-1, 2)
        .astype(int)
    )
    transformed_xmin = np.min(transformed_points[:, 0])
    transformed_ymin = np.min(transformed_points[:, 1])
    transformed_xmax = np.max(transformed_points[:, 0])
    transformed_ymax = np.max(transformed_points[:, 1])
    if not allow_negative_and_large_coords:
        transformed_xmin = max(0, min(transformed_xmin, result_width - 1))
        transformed_ymin = max(0, min(transformed_ymin, result_height - 1))
        transformed_xmax = max(0, min(transformed_xmax, result_width - 1))
        transformed_ymax = max(0, min(transformed_ymax, result_height - 1))
    if remove_bad_coords and not (
        transformed_xmin >= 0
        and transformed_ymin >= 0
        and transformed_xmax < result_width
        and transformed_ymax < result_height
        and transformed_xmin < transformed_xmax
        and transformed_ymin < transformed_ymax
    ):
        return None

    transformed_bbox_data = copy.deepcopy(bbox_data)
    transformed_bbox_data.xmin = transformed_xmin
    transformed_bbox_data.ymin = transformed_ymin
    transformed_bbox_data.xmax = transformed_xmax
    transformed_bbox_data.ymax = transformed_ymax
    transformed_bbox_data.keypoints = apply_perspective_transform_to_points(
        transformed_bbox_data.keypoints,
        perspective_matrix,
        result_height,
        result_height,
        allow_negative_and_large_coords,
        remove_bad_coords,
    )
    if isinstance(transformed_bbox_data.mask, list):
        transformed_bbox_data.mask = [
            apply_perspective_transform_to_points(
                polygon,
                perspective_matrix,
                result_height,
                result_height,
                allow_negative_and_large_coords,
                remove_bad_coords,
            )
            for polygon in transformed_bbox_data.mask
        ]
        transformed_bbox_data.mask = [polygon for polygon in transformed_bbox_data.mask if len(polygon) > 0]
    transformed_bbox_data.additional_bboxes_data = [
        _apply_perspective_transform_to_bbox_data(
            additional_bbox_data,
            perspective_matrix,
            result_height,
            result_height,
            allow_negative_and_large_coords,
            remove_bad_coords,
        )
        for additional_bbox_data in transformed_bbox_data.additional_bboxes_data
    ]
    transformed_bbox_data.additional_bboxes_data = [
        additional_bbox_data
        for additional_bbox_data in transformed_bbox_data.additional_bboxes_data
        if additional_bbox_data is not None
    ]
    transformed_bbox_data.cropped_image = None
    return transformed_bbox_data


def get_perspective_matrix_for_base_keypoints(
    base_keypoints: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    base_keypoints = np.array(base_keypoints, dtype=np.float32)
    (top_left, top_right, bottom_right, bottom_left) = base_keypoints
    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    result_width = max(int(width_a), int(width_b))
    result_height = max(int(height_a), int(height_b))
    transformed_points = np.array(
        [[0, 0], [result_width - 1, 0], [result_width - 1, result_height - 1], [0, result_height - 1]], dtype=np.float32
    )
    perspective_matrix = cv2.getPerspectiveTransform(base_keypoints, transformed_points)
    return perspective_matrix, (result_width, result_height)


def apply_perspective_transform_to_image_data(
    image_data: ImageData,
    perspective_matrix: np.ndarray,
    result_width: int,
    result_height: int,
    allow_negative_and_large_coords: bool,
    remove_bad_coords: bool,
    open_image: bool = True,
    exif_transpose: bool = False,
) -> ImageData:
    image = image_data.open_image(returns_none_if_empty=True, exif_transpose=exif_transpose) if open_image else None
    image = cv2.warpPerspective(image, perspective_matrix, (result_width, result_height)) if image is not None else None

    image_data = copy.deepcopy(image_data)
    image_data.keypoints = apply_perspective_transform_to_points(
        image_data.keypoints,
        perspective_matrix,
        result_width,
        result_height,
        allow_negative_and_large_coords,
        remove_bad_coords,
    )
    if isinstance(image_data.mask, list):
        image_data.mask = [
            apply_perspective_transform_to_points(
                polygon,
                perspective_matrix,
                result_height,
                result_height,
                allow_negative_and_large_coords,
                remove_bad_coords,
            )
            for polygon in image_data.mask
        ]
        image_data.mask = [polygon for polygon in image_data.mask if len(polygon) > 0]
    transformed_bboxes_data = [
        _apply_perspective_transform_to_bbox_data(
            bbox_data,
            perspective_matrix,
            result_width,
            result_height,
            allow_negative_and_large_coords,
            remove_bad_coords,
        )
        for bbox_data in image_data.bboxes_data
    ]
    image_data.bboxes_data = [bbox_data for bbox_data in transformed_bboxes_data if bbox_data is not None]
    image_data.image_path = None
    image_data.image = image

    return image_data


def non_max_suppression_image_data(
    image_data: ImageData, iou: float, score_threshold: float = float("-inf")
) -> ImageData:
    """
    Taken from https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5
    """
    image_data = copy.deepcopy(image_data)
    boxes = np.array([bbox_data.coords for bbox_data in image_data.bboxes_data])
    if len(boxes) == 0:
        return image_data
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(boxes))
    for i, bbox_data in enumerate(image_data.bboxes_data):
        temp_indices = indices[indices != i]
        xx1 = np.maximum(bbox_data.xmin, boxes[temp_indices, 0])
        yy1 = np.maximum(bbox_data.ymin, boxes[temp_indices, 1])
        xx2 = np.minimum(bbox_data.xmax, boxes[temp_indices, 2])
        yy2 = np.minimum(bbox_data.ymax, boxes[temp_indices, 3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > iou:
            indices = indices[indices != i]

    image_data.bboxes_data = [image_data.bboxes_data[i] for i in indices]
    if score_threshold is not None:
        image_data.bboxes_data = [
            bbox_data
            for bbox_data in image_data.bboxes_data
            if (bbox_data.detection_score is None) or (bbox_data.detection_score >= score_threshold)
        ]
    image_data.image_path = image_data.image_path
    image_data.image = image_data.image
    return image_data


def non_max_suppression_image_data_using_tf(
    image_data: ImageData, iou: float, score_threshold: float = float("-inf"), ignore_classes: bool = True
) -> ImageData:
    import tensorflow as tf

    image_data = copy.deepcopy(image_data)
    if len(image_data.bboxes_data) <= 1:
        return image_data
    if not ignore_classes:
        labels = sorted(set([bbox_data.label for bbox_data in image_data.bboxes_data]))
        indexes = []
        for label in labels:
            indexes_by_label = [i for i, bbox_data in enumerate(image_data.bboxes_data) if bbox_data.label == label]
            bboxes_data_by_label = [bbox_data for bbox_data in image_data.bboxes_data if bbox_data.label == label]
            bboxes_by_label = [
                (bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax) for bbox_data in bboxes_data_by_label
            ]
            scores_by_label = [
                bbox_data.detection_score if bbox_data.detection_score is not None else 1.0
                for bbox_data in bboxes_data_by_label
            ]
            result_by_label = tf.image.non_max_suppression(
                bboxes_by_label,
                scores_by_label,
                len(bboxes_data_by_label),
                iou_threshold=float(iou),
                score_threshold=float(score_threshold),
            )
            indexes.extend([indexes_by_label[i] for i in result_by_label.numpy()])
        image_data.bboxes_data = [image_data.bboxes_data[i] for i in sorted(indexes)]
    else:
        bboxes = [
            (bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax) for bbox_data in image_data.bboxes_data
        ]
        scores = [
            bbox_data.detection_score if bbox_data.detection_score is not None else 1.0
            for bbox_data in image_data.bboxes_data
        ]
        result = tf.image.non_max_suppression(
            bboxes,
            scores,
            len(image_data.bboxes_data),
            iou_threshold=float(iou),
            score_threshold=float(score_threshold),
        )
        image_data.bboxes_data = [image_data.bboxes_data[i] for i in result.numpy()]
    return image_data


def split_by_grid(
    size: Tuple[int, int],  # (width, height)
    n_rows: int,
    n_cols: int,
    x_window_size: int,
    y_window_size: int,
    x_offset: int,
    y_offset: int,
    minimum_size: float = 0.5,
) -> List[BboxData]:
    width, height = size
    bboxes_data = [
        BboxData(
            xmin=x_offset + i * x_window_size,
            ymin=y_offset + j * y_window_size,
            xmax=min(width, x_offset + (i + 1) * x_window_size),
            ymax=min(height, y_offset + (j + 1) * y_window_size),
        )
        for i in range(n_rows)
        for j in range(n_cols)
    ]
    bboxes_data = [
        bbox_data
        for bbox_data in bboxes_data
        if (
            (bbox_data.xmax - bbox_data.xmin >= minimum_size * x_window_size)
            and (bbox_data.ymax - bbox_data.ymin >= minimum_size * y_window_size)
        )
    ]
    return bboxes_data


def split_image_by_grid(
    image: np.ndarray,
    n_rows: int,
    n_cols: int,
    x_window_size: int,
    y_window_size: int,
    x_offset: int,
    y_offset: int,
    minimum_size: float = 0.5,
) -> List[BboxData]:
    height, width = image.shape[0:2]
    bboxes_data = split_by_grid(
        size=(width, height),  # (width, height)
        n_rows=n_rows,
        n_cols=n_cols,
        x_window_size=x_window_size,
        y_window_size=y_window_size,
        x_offset=x_offset,
        y_offset=y_offset,
        minimum_size=minimum_size,
    )
    for bbox_data in bboxes_data:
        bbox_data.image = image
    return bboxes_data


def split_image_data_by_grid(
    image_data: ImageData,
    n_rows: int,
    n_cols: int,
    x_window_size: int,
    y_window_size: int,
    x_offset: int,
    y_offset: int,
    remove_bad_coords: bool,
    minimum_size: float = 0.5,
) -> ImageData:
    image_data = copy.deepcopy(image_data)
    width, height = image_data.get_image_size()
    crops_bboxes_data = split_by_grid(
        size=(width, height),
        n_rows=n_rows,
        n_cols=n_cols,
        x_window_size=x_window_size,
        y_window_size=y_window_size,
        x_offset=x_offset,
        y_offset=y_offset,
        minimum_size=minimum_size,
    )

    def if_bbox_data_inside_crop(crop_bbox_data: BboxData, bbox_data: BboxData):
        bbox_data.keypoints = bbox_data.keypoints[
            (
                (bbox_data.keypoints[:, 0] >= crop_bbox_data.xmin)
                & (bbox_data.keypoints[:, 1] >= crop_bbox_data.ymin)
                & (bbox_data.keypoints[:, 0] <= crop_bbox_data.ymax)
                & (bbox_data.keypoints[:, 1] <= crop_bbox_data.ymax)
            )
        ]
        bbox_data.additional_bboxes_data = [
            additional_bbox_data
            for additional_bbox_data in bbox_data.additional_bboxes_data
            if if_bbox_data_inside_crop(crop_bbox_data, additional_bbox_data)
        ]
        return (
            bbox_data.xmin >= crop_bbox_data.xmin
            and bbox_data.ymin >= crop_bbox_data.ymin
            and bbox_data.xmax <= crop_bbox_data.xmax
            and bbox_data.ymax <= crop_bbox_data.ymax
            and bbox_data.xmin < bbox_data.xmax
            and bbox_data.ymin < bbox_data.ymax
        )

    for crop_bbox_data in crops_bboxes_data:
        if remove_bad_coords:
            additional_bboxes_data = [
                bbox_data
                for bbox_data in copy.deepcopy(image_data.bboxes_data)
                if if_bbox_data_inside_crop(crop_bbox_data, bbox_data)
            ]
            keypoints = copy.deepcopy(image_data.keypoints)[
                (
                    (image_data.keypoints[:, 0] >= crop_bbox_data.xmin)
                    & (image_data.keypoints[:, 1] >= crop_bbox_data.ymin)
                    & (image_data.keypoints[:, 0] <= crop_bbox_data.ymax)
                    & (image_data.keypoints[:, 1] <= crop_bbox_data.ymax)
                )
            ]
        else:
            additional_bboxes_data = copy.deepcopy(image_data.bboxes_data)
            keypoints = copy.deepcopy(image_data.keypoints)

        crop_bbox_data.additional_bboxes_data = additional_bboxes_data
        crop_bbox_data.keypoints = keypoints

    image_data.bboxes_data = crops_bboxes_data
    image_data.image_path = image_data.image_path  # apply to bboxes_data
    image_data.image = image_data.image  # apply to bboxes_data
    return image_data


def uncrop_bboxes_data(
    bboxes_data: List[BboxData],
    src_xmin: int,
    src_ymin: int,
    src_image: Optional[np.ndarray] = None,
    src_image_path: Optional[np.ndarray] = None,
    src_image_height: Optional[int] = None,
    src_image_width: Optional[int] = None,
) -> BboxData:
    bboxes_data = copy.deepcopy(bboxes_data)

    def _append_cropped_bbox_data_to_image_data(bbox_data: BboxData):
        bbox_data.keypoints[:, 0] += src_xmin
        bbox_data.keypoints[:, 1] += src_ymin
        bbox_data.xmin += src_xmin
        bbox_data.ymin += src_ymin
        bbox_data.xmax += src_xmin
        bbox_data.ymax += src_ymin
        bbox_data.image = src_image
        bbox_data.image_path = src_image_path
        bbox_data.cropped_image = None
        bbox_data.meta_height = src_image_height
        bbox_data.meta_width = src_image_width
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            _append_cropped_bbox_data_to_image_data(additional_bbox_data)

    for bbox_data in bboxes_data:
        _append_cropped_bbox_data_to_image_data(bbox_data)

    return bboxes_data


def concat_images_data(
    image_data_a: ImageData,
    image_data_b: ImageData,
    background_color_a: Tuple[int, int, int, int] = None,
    background_color_b: Tuple[int, int, int, int] = None,
    thumbnail_size_a: Tuple[int, int] = None,
    thumbnail_size_b: Tuple[int, int] = None,
    how: Literal["horizontally", "vertically"] = "horizontally",
    mode: Literal["L", "RGB", "RGBA"] = "RGBA",
    background_edge_width: int = 3,
    between_edge_width: int = 0,
    exif_transpose: bool = False,
) -> ImageData:
    image_data_a = copy.deepcopy(image_data_a)
    image_data_b = copy.deepcopy(image_data_b)

    if image_data_a is None and image_data_b is not None:
        return image_data_b
    if image_data_a is not None and image_data_b is None:
        return image_data_a

    image_a = image_data_a.open_image(exif_transpose=exif_transpose)
    image_b = image_data_b.open_image(exif_transpose=exif_transpose)

    ha, wa = image_a.shape[:2]
    hb, wb = image_b.shape[:2]

    image = concat_images(
        image_a=image_a,
        image_b=image_b,
        background_color_a=background_color_a,
        background_color_b=background_color_b,
        thumbnail_size_a=thumbnail_size_a,
        thumbnail_size_b=thumbnail_size_b,
        how=how,
        mode=mode,
        background_edge_width=background_edge_width,
        between_edge_width=between_edge_width,
    )
    image_data_a_new_xmin, image_data_a_new_ymin = None, None
    image_data_b_new_xmin, image_data_b_new_ymin = None, None

    if how == "horizontally":
        max_height = np.max([ha, hb])
        min_ha = max_height // 2 - ha // 2
        max_ha = max_height // 2 + ha // 2
        min_hb = max_height // 2 - hb // 2
        max_hb = max_height // 2 + hb // 2
        image_data_a_new_xmin = 0
        image_data_a_new_ymin = min_ha
        image_data_a_new_xmax = wa
        image_data_a_new_ymax = max_ha
        image_data_b_new_xmin = wa + between_edge_width
        image_data_b_new_ymin = min_hb
        image_data_b_new_xmax = wa + between_edge_width + wb
        image_data_b_new_ymax = max_hb

    elif how == "vertically":
        max_width = np.max([wa, wb])
        min_wa = max_width // 2 - wa // 2
        max_wa = max_width // 2 + wa // 2
        min_wb = max_width // 2 - wb // 2
        max_wb = max_width // 2 + wb // 2
        image_data_a_new_xmin = min_wa
        image_data_a_new_ymin = 0
        image_data_a_new_xmax = max_wa
        image_data_a_new_ymax = ha
        image_data_b_new_xmin = min_wb
        image_data_b_new_ymin = ha + between_edge_width
        image_data_b_new_xmax = max_wb
        image_data_b_new_ymax = ha + between_edge_width + hb

    keypoints_a = image_data_a.keypoints
    keypoints_b = image_data_a.keypoints
    keypoints_a[:, 0] += image_data_a_new_xmin
    keypoints_a[:, 1] += image_data_a_new_ymin
    keypoints_b[:, 0] += image_data_b_new_xmin
    keypoints_b[:, 1] += image_data_b_new_ymin

    def _get_new_coords_for_bbox_data(bbox_data: BboxData, xmin: int, ymin: int):
        bbox_data.keypoints[:, 0] += xmin
        bbox_data.keypoints[:, 1] += ymin
        bbox_data.xmin += xmin
        bbox_data.ymin += ymin
        bbox_data.xmax += xmin
        bbox_data.ymax += ymin
        bbox_data.image = None
        bbox_data.image_path = None
        bbox_data.cropped_image = None
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            _get_new_coords_for_bbox_data(additional_bbox_data, xmin, ymin)

    for bbox_data in image_data_a.bboxes_data:
        _get_new_coords_for_bbox_data(bbox_data, image_data_a_new_xmin, image_data_a_new_ymin)

    if "concat_images_data__image_data" not in [bbox_data.label for bbox_data in image_data_a.bboxes_data]:
        bbox_data_a_into = [
            BboxData(
                xmin=image_data_a_new_xmin,
                ymin=image_data_a_new_ymin,
                xmax=image_data_a_new_xmax,
                ymax=image_data_a_new_ymax,
                label="concat_images_data__image_data",
                additional_bboxes_data=[
                    bbox_data
                    for bbox_data in image_data_a.bboxes_data
                    if "concat_images_data__image_data" != bbox_data.label
                ],
            )
        ]
    else:
        bbox_data_a_into = []
    image_data_a.bboxes_data = [
        bbox_data for bbox_data in image_data_a.bboxes_data if "concat_images_data__image_data" == bbox_data.label
    ] + bbox_data_a_into

    for bbox_data in image_data_b.bboxes_data:
        _get_new_coords_for_bbox_data(bbox_data, image_data_b_new_xmin, image_data_b_new_ymin)
    if "concat_images_data__image_data" not in [bbox_data.label for bbox_data in image_data_b.bboxes_data]:
        bbox_data_b_into = [
            BboxData(
                xmin=image_data_b_new_xmin,
                ymin=image_data_b_new_ymin,
                xmax=image_data_b_new_xmax,
                ymax=image_data_b_new_ymax,
                label="concat_images_data__image_data",
                additional_bboxes_data=[
                    bbox_data
                    for bbox_data in image_data_b.bboxes_data
                    if "concat_images_data__image_data" != bbox_data.label
                ],
            )
        ]
    else:
        bbox_data_b_into = []
    image_data_b.bboxes_data = [
        bbox_data for bbox_data in image_data_b.bboxes_data if "concat_images_data__image_data" == bbox_data.label
    ] + bbox_data_b_into

    image_data = ImageData(
        image_path=None,
        image=image,
        bboxes_data=image_data_a.bboxes_data + image_data_b.bboxes_data,
        label=None,
        keypoints=np.concatenate([keypoints_a, keypoints_b], axis=0),
        additional_info={**image_data_a.additional_info, **image_data_b.additional_info},
    )

    return image_data


def get_all_bboxes_data_in_image_data(
    image_data: ImageData,
    additional_bboxes_data_depth: Optional[int] = None,
) -> List[BboxData]:
    bboxes_data = []

    def _append_bbox_data(bbox_data: BboxData, depth: int):
        if additional_bboxes_data_depth is not None and depth > additional_bboxes_data_depth:
            return
        bboxes_data.append(bbox_data)
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            _append_bbox_data(additional_bbox_data, depth + 1)

    for bbox_data in image_data.bboxes_data:
        _append_bbox_data(bbox_data, depth=0)

    return bboxes_data


def flatten_additional_bboxes_data_in_image_data(
    image_data: ImageData,
    additional_bboxes_data_depth: Optional[int] = None,
    set_additional_bboxes_data_empty: bool = True,
) -> ImageData:
    image_data = copy.deepcopy(image_data)
    bboxes_data = []

    def _append_bbox_data(bbox_data: BboxData, depth: int):
        if additional_bboxes_data_depth is not None and depth > additional_bboxes_data_depth:
            return
        bboxes_data.append(bbox_data)
        for additional_bbox_data in bbox_data.additional_bboxes_data:
            _append_bbox_data(additional_bbox_data, depth + 1)
        if set_additional_bboxes_data_empty:
            bbox_data.additional_bboxes_data = []

    for bbox_data in image_data.bboxes_data:
        _append_bbox_data(bbox_data, depth=0)

    image_data.bboxes_data = bboxes_data
    return image_data


def find_closest_pair(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    distances = np.sum((arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :]) ** 2, axis=-1)
    return np.unravel_index(np.argmin(distances, axis=None), distances.shape)


def combine_polygons(polygons: List[np.ndarray]) -> np.ndarray:
    combined = []
    polygons = [np.array(polygon).reshape(-1, 2) for polygon in polygons]
    index_pairs = [[] for _ in range(len(polygons))]

    for i in range(1, len(polygons)):
        idx1, idx2 = find_closest_pair(polygons[i - 1], polygons[i])
        index_pairs[i - 1].append(idx1)
        index_pairs[i].append(idx2)

    for pass_num in range(2):
        if pass_num == 0:
            for i, indices in enumerate(index_pairs):
                if len(indices) == 2 and indices[0] > indices[1]:
                    indices = indices[::-1]
                    polygons[i] = polygons[i][::-1, :]

                polygons[i] = np.roll(polygons[i], -indices[0], axis=0)
                polygons[i] = np.concatenate([polygons[i], polygons[i][:1]])
                if i in [0, len(index_pairs) - 1]:
                    combined.append(polygons[i])
                else:
                    indices = [0, indices[1] - indices[0]]
                    combined.append(polygons[i][indices[0] : indices[1] + 1])
        else:
            for i in range(len(index_pairs) - 1, -1, -1):
                if i not in [0, len(index_pairs) - 1]:
                    indices = index_pairs[i]
                    adjusted_idx = abs(indices[1] - indices[0])
                    combined.append(polygons[i][adjusted_idx:])
    return combined


def combine_mask_polygons_to_one_polygon(mask: List[List[np.ndarray]]) -> np.ndarray:
    if len(mask) > 1:
        keypoints = np.concatenate(combine_polygons(mask), axis=0).reshape(-1, 2)
    elif len(mask) == 1:
        keypoints = mask[0]
    else:
        keypoints = []
    keypoints = np.array(keypoints).reshape(-1, 2)
    return keypoints
