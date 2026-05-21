import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.utils.images_datas import (
    apply_perspective_transform_to_image_data,
    crop_image_data,
    flatten_additional_bboxes_data_in_image_data,
    get_all_bboxes_data_in_image_data,
    get_image_data_filtered_by_labels,
    get_n_bboxes_data_filtered_by_labels,
    non_max_suppression_image_data,
    resize_image_data,
    rotate_image_data,
    split_by_grid,
)


def _all_bboxes(bboxes_data):
    result = []
    for bbox_data in bboxes_data:
        result.append(bbox_data)
        result.extend(_all_bboxes(bbox_data.additional_bboxes_data))
    return result


def test_label_filtering_for_image_data_and_bbox_groups():
    cat = BboxData(xmin=0, ymin=0, xmax=2, ymax=2, label="cat")
    dog = BboxData(xmin=2, ymin=2, xmax=4, ymax=4, label="dog")
    image_data = ImageData(image=np.zeros((5, 5, 3), dtype=np.uint8), bboxes_data=[cat, dog])

    included = get_image_data_filtered_by_labels(image_data, ["cat"])
    excluded = get_n_bboxes_data_filtered_by_labels([[cat, dog]], ["cat"], include=False)

    assert [bbox_data.label for bbox_data in included.bboxes_data] == ["cat"]
    assert [bbox_data.label for bbox_data in excluded[0]] == ["dog"]


def test_resize_crop_and_rotate90_transform_bbox_keypoints_and_mask():
    image_data = ImageData(
        image=np.zeros((10, 20, 3), dtype=np.uint8),
        keypoints=[(10, 5)],
        bboxes_data=[
            BboxData(
                xmin=2,
                ymin=2,
                xmax=8,
                ymax=6,
                label="object",
                keypoints=[(4, 4)],
                mask=[[(2, 2), (8, 2), (8, 6), (2, 6)]],
            )
        ],
    )

    resized = resize_image_data(image_data, size=(40, 20))
    cropped = crop_image_data(resized, xmin=0, ymin=0, xmax=20, ymax=10, allow_negative_and_large_coords=False, remove_bad_coords=False)
    rotated = rotate_image_data(cropped, angle=90)

    assert resized.image.shape == (20, 40, 3)
    assert resized.bboxes_data[0].coords == (4, 4, 16, 12)
    assert cropped.bboxes_data[0].coords == (4, 4, 16, 9)
    assert rotated.image.shape == (20, 10, 3)
    assert rotated.bboxes_data[0].coords == (4, 4, 9, 16)


def test_non_max_suppression_filters_overlapping_and_low_score_boxes():
    image_data = ImageData(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        bboxes_data=[
            BboxData(xmin=0, ymin=0, xmax=10, ymax=10, detection_score=0.9, label="a"),
            BboxData(xmin=1, ymin=1, xmax=9, ymax=9, detection_score=0.8, label="b"),
            BboxData(xmin=12, ymin=12, xmax=18, ymax=18, detection_score=0.1, label="c"),
        ],
    )

    result = non_max_suppression_image_data(image_data, iou=0.5, score_threshold=0.5)

    assert [bbox_data.label for bbox_data in result.bboxes_data] == ["b"]


def test_split_by_grid_and_recursive_bbox_collection():
    child = BboxData(xmin=1, ymin=1, xmax=2, ymax=2, label="child")
    parent = BboxData(xmin=0, ymin=0, xmax=4, ymax=4, label="parent", additional_bboxes_data=[child])
    image_data = ImageData(image=np.zeros((8, 8, 3), dtype=np.uint8), bboxes_data=[parent])

    grid = split_by_grid(size=(8, 8), n_rows=2, n_cols=2, x_window_size=4, y_window_size=4, x_offset=0, y_offset=0)
    all_bboxes = get_all_bboxes_data_in_image_data(image_data)

    assert [bbox.coords for bbox in grid] == [(0, 0, 4, 4), (0, 4, 4, 8), (4, 0, 8, 4), (4, 4, 8, 8)]
    assert [bbox.label for bbox in all_bboxes] == ["parent", "child"]


def test_image_data_transforms_do_not_duplicate_source_image_across_bboxes():
    source_image = np.zeros((64, 80, 3), dtype=np.uint8)
    image_data = ImageData(
        image=source_image,
        keypoints=[(10, 10)],
        bboxes_data=[
            BboxData(
                xmin=5,
                ymin=5,
                xmax=30,
                ymax=30,
                additional_bboxes_data=[
                    BboxData(xmin=8, ymin=8, xmax=15, ymax=15),
                    BboxData(xmin=16, ymin=16, xmax=22, ymax=22),
                ],
            )
            for _ in range(20)
        ],
    )

    resized = resize_image_data(image_data, size=(40, 32))
    cropped = crop_image_data(image_data, xmin=0, ymin=0, xmax=40, ymax=32, allow_negative_and_large_coords=False, remove_bad_coords=False)
    rotated = rotate_image_data(image_data, angle=90)
    perspective = apply_perspective_transform_to_image_data(
        image_data,
        perspective_matrix=np.eye(3, dtype=np.float32),
        result_width=80,
        result_height=64,
        allow_negative_and_large_coords=False,
        remove_bad_coords=False,
    )

    for transformed in [resized, cropped, rotated, perspective]:
        assert transformed.image is not source_image
        transformed_bboxes = _all_bboxes(transformed.bboxes_data)
        assert len(transformed_bboxes) == 60
        assert {id(bbox_data.image) for bbox_data in transformed_bboxes} == {id(transformed.image)}
        assert all(bbox_data.image is transformed.image for bbox_data in transformed_bboxes)

    original_bboxes = _all_bboxes(image_data.bboxes_data)
    assert {id(bbox_data.image) for bbox_data in original_bboxes} == {id(source_image)}
    assert all(bbox_data.image is source_image for bbox_data in original_bboxes)


def test_logical_helpers_deepcopy_structure_without_copying_image_arrays():
    source_image = np.zeros((32, 32, 3), dtype=np.uint8)
    image_data = ImageData(
        image=source_image,
        bboxes_data=[
            BboxData(
                xmin=0,
                ymin=0,
                xmax=10,
                ymax=10,
                label="keep",
                detection_score=0.9,
                additional_bboxes_data=[BboxData(xmin=1, ymin=1, xmax=5, ymax=5, label="child")],
            ),
            BboxData(xmin=20, ymin=20, xmax=30, ymax=30, label="drop", detection_score=0.1),
        ],
    )

    filtered = get_image_data_filtered_by_labels(image_data, ["keep"])
    nms = non_max_suppression_image_data(image_data, iou=0.5, score_threshold=0.5)
    flattened = flatten_additional_bboxes_data_in_image_data(image_data)

    for result in [filtered, nms, flattened]:
        assert result is not image_data
        result_bboxes = _all_bboxes(result.bboxes_data)
        assert {id(bbox_data.image) for bbox_data in result_bboxes} == {id(result.image)}
        assert result.image is source_image
        assert all(bbox_data.image is source_image for bbox_data in result_bboxes)
