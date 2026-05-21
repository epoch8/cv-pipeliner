import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.utils.images_datas import (
    apply_perspective_transform_to_image_data,
    apply_perspective_transform_to_points,
    concat_images_data,
    crop_image_data,
    flatten_additional_bboxes_data_in_image_data,
    get_all_bboxes_data_in_image_data,
    get_image_data_filtered_by_labels,
    get_n_bboxes_data_filtered_by_labels,
    non_max_suppression_image_data,
    resize_image_data,
    rotate_image_data,
    split_image_data_by_grid,
    split_by_grid,
    thumbnail_image_data,
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


def test_resize_and_thumbnail_preserve_metadata_for_annotation_only_image_data():
    image_data = ImageData(
        meta_width=20,
        meta_height=10,
        keypoints=[(10, 5)],
        mask=[[(0, 0), (19, 0), (19, 9), (0, 9)]],
        bboxes_data=[
            BboxData(
                xmin=2,
                ymin=2,
                xmax=8,
                ymax=6,
                keypoints=[(4, 4)],
                mask=[[(2, 2), (8, 2), (8, 6), (2, 6)]],
                additional_bboxes_data=[
                    BboxData(xmin=3, ymin=3, xmax=6, ymax=5, keypoints=[(4, 4)]),
                ],
            )
        ],
    )

    resized = resize_image_data(image_data, size=(40, 20), open_image=False)
    thumbnail = thumbnail_image_data(image_data, size=10)

    assert resized.image is None
    assert resized.get_image_size() == (40, 20)
    assert resized.bboxes_data[0].get_image_size() == (40, 20)
    assert resized.bboxes_data[0].additional_bboxes_data[0].get_image_size() == (40, 20)
    assert resized.bboxes_data[0].coords == (4, 4, 16, 12)
    np.testing.assert_allclose(resized.bboxes_data[0].coords_n, (0.1, 0.2, 0.4, 0.6))
    np.testing.assert_array_equal(resized.keypoints, np.array([[20, 10]]))
    np.testing.assert_array_equal(resized.bboxes_data[0].keypoints, np.array([[8, 8]]))

    assert thumbnail.image is None
    assert thumbnail.get_image_size() == (10, 5)
    assert thumbnail.bboxes_data[0].get_image_size() == (10, 5)
    assert thumbnail.bboxes_data[0].coords == (1, 1, 4, 3)
    np.testing.assert_allclose(thumbnail.bboxes_data[0].coords_n, (0.1, 0.2, 0.4, 0.6))


def test_rotate_image_data_transforms_top_level_mask_and_preserves_keypoints():
    image_data = ImageData(
        meta_width=100,
        meta_height=50,
        keypoints=[(80, 40)],
        mask=[[(70, 30), (90, 30), (90, 45), (70, 45)]],
        bboxes_data=[
            BboxData(
                xmin=70,
                ymin=30,
                xmax=90,
                ymax=45,
                keypoints=[(80, 40)],
                mask=[[(70, 30), (90, 30), (90, 45), (70, 45)]],
            )
        ],
    )

    rotated = rotate_image_data(image_data, angle=90, open_image=False)

    assert rotated.image is None
    assert rotated.get_image_size() == (50, 100)
    np.testing.assert_array_equal(rotated.keypoints, np.array([[40, 19]]))
    np.testing.assert_array_equal(rotated.mask[0], np.array([[30, 29], [30, 9], [45, 9], [45, 29]]))
    assert rotated.bboxes_data[0].coords == (30, 10, 45, 30)
    np.testing.assert_array_equal(rotated.bboxes_data[0].keypoints, np.array([[40, 19]]))
    np.testing.assert_array_equal(
        rotated.bboxes_data[0].mask[0],
        np.array([[30, 29], [30, 9], [45, 9], [45, 29]]),
    )


def test_perspective_identity_keeps_non_square_annotation_coordinates_and_metadata():
    image_data = ImageData(
        meta_width=100,
        meta_height=50,
        keypoints=[(80, 40)],
        mask=[[(70, 30), (90, 30), (90, 45), (70, 45)]],
        bboxes_data=[
            BboxData(
                xmin=70,
                ymin=30,
                xmax=90,
                ymax=45,
                keypoints=[(80, 40)],
                mask=[[(70, 30), (90, 30), (90, 45), (70, 45)]],
                additional_bboxes_data=[
                    BboxData(xmin=75, ymin=35, xmax=85, ymax=40, keypoints=[(80, 38)]),
                ],
            )
        ],
    )

    transformed = apply_perspective_transform_to_image_data(
        image_data,
        perspective_matrix=np.eye(3, dtype=np.float32),
        result_width=100,
        result_height=50,
        allow_negative_and_large_coords=False,
        remove_bad_coords=True,
        open_image=False,
    )

    assert transformed.image is None
    assert transformed.get_image_size() == (100, 50)
    np.testing.assert_array_equal(transformed.keypoints, np.array([[80, 40]]))
    np.testing.assert_array_equal(transformed.mask[0], np.array([[70, 30], [90, 30], [90, 45], [70, 45]]))
    assert transformed.bboxes_data[0].coords == (70, 30, 90, 45)
    assert transformed.bboxes_data[0].get_image_size() == (100, 50)
    np.testing.assert_array_equal(transformed.bboxes_data[0].keypoints, np.array([[80, 40]]))
    np.testing.assert_array_equal(
        transformed.bboxes_data[0].mask[0],
        np.array([[70, 30], [90, 30], [90, 45], [70, 45]]),
    )
    assert transformed.bboxes_data[0].additional_bboxes_data[0].coords == (75, 35, 85, 40)


def test_crop_image_data_updates_annotation_only_metadata_and_shifts_masks():
    image_data = ImageData(
        meta_width=100,
        meta_height=50,
        keypoints=[(80, 40), (10, 10)],
        mask=[[(70, 30), (90, 30), (90, 45), (70, 45)]],
        bboxes_data=[
            BboxData(
                xmin=70,
                ymin=30,
                xmax=90,
                ymax=45,
                keypoints=[(80, 40)],
                mask=[[(70, 30), (90, 30), (90, 45), (70, 45)]],
            )
        ],
    )

    cropped = crop_image_data(
        image_data,
        xmin=60,
        ymin=20,
        xmax=99,
        ymax=49,
        allow_negative_and_large_coords=False,
        remove_bad_coords=True,
        open_image=False,
    )

    assert cropped.image is None
    assert cropped.get_image_size() == (39, 29)
    np.testing.assert_array_equal(cropped.keypoints, np.array([[20, 20]]))
    np.testing.assert_array_equal(cropped.mask[0], np.array([[10, 10], [30, 10], [30, 25], [10, 25]]))
    assert cropped.bboxes_data[0].coords == (10, 10, 30, 25)
    assert cropped.bboxes_data[0].get_image_size() == (39, 29)
    np.testing.assert_array_equal(cropped.bboxes_data[0].keypoints, np.array([[20, 20]]))
    np.testing.assert_array_equal(cropped.bboxes_data[0].mask[0], np.array([[10, 10], [30, 10], [30, 25], [10, 25]]))


def test_crop_image_data_clips_partially_visible_masks_without_dropping_edges():
    image_data = ImageData(
        meta_width=100,
        meta_height=50,
        mask=[[(40, 10), (80, 10), (80, 40), (40, 40)]],
        bboxes_data=[
            BboxData(
                xmin=40,
                ymin=10,
                xmax=80,
                ymax=40,
                mask=[[(40, 10), (80, 10), (80, 40), (40, 40)]],
            )
        ],
    )

    cropped = crop_image_data(
        image_data,
        xmin=60,
        ymin=20,
        xmax=99,
        ymax=49,
        allow_negative_and_large_coords=False,
        remove_bad_coords=True,
        open_image=False,
    )

    expected_mask = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
    np.testing.assert_array_equal(cropped.mask[0], expected_mask)
    np.testing.assert_array_equal(cropped.bboxes_data[0].mask[0], expected_mask)
    assert cropped.bboxes_data[0].coords == (0, 0, 20, 20)


def test_split_image_data_by_grid_filters_keypoints_using_crop_xmax_not_ymax():
    image_data = ImageData(
        meta_width=100,
        meta_height=50,
        keypoints=[(75, 25), (25, 25)],
        bboxes_data=[
            BboxData(xmin=70, ymin=10, xmax=90, ymax=40, keypoints=[(75, 25)]),
        ],
    )

    split = split_image_data_by_grid(
        image_data,
        n_rows=1,
        n_cols=2,
        x_window_size=50,
        y_window_size=50,
        x_offset=0,
        y_offset=0,
        minimum_crop_intersection_area=0.2,
        minimum_relative_size_of_inner_bboxes=0.2,
    )

    assert [bbox.coords for bbox in split.bboxes_data] == [(0, 0, 50, 50), (50, 0, 100, 50)]
    np.testing.assert_array_equal(split.bboxes_data[0].keypoints, np.array([[25, 25]]))
    assert split.bboxes_data[0].additional_bboxes_data == []
    np.testing.assert_array_equal(split.bboxes_data[1].keypoints, np.array([[75, 25]]))
    assert split.bboxes_data[1].additional_bboxes_data[0].coords == (70, 10, 90, 40)
    np.testing.assert_array_equal(split.bboxes_data[1].additional_bboxes_data[0].keypoints, np.array([[75, 25]]))


def test_apply_perspective_transform_to_points_clips_transformed_points():
    perspective_matrix = np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    transformed = apply_perspective_transform_to_points(
        points=[(80, 10)],
        perspective_matrix=perspective_matrix,
        result_width=100,
        result_height=50,
        allow_negative_and_large_coords=False,
        remove_bad_coords=False,
    )

    np.testing.assert_array_equal(transformed, np.array([[99, 10]]))


def test_concat_images_data_offsets_keypoints_from_both_images():
    image_data_a = ImageData(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        keypoints=[(1, 1)],
        bboxes_data=[BboxData(xmin=1, ymin=1, xmax=4, ymax=4, keypoints=[(2, 2)], label="a")],
    )
    image_data_b = ImageData(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        keypoints=[(2, 2)],
        bboxes_data=[BboxData(xmin=2, ymin=2, xmax=5, ymax=5, keypoints=[(3, 3)], label="b")],
    )

    concatenated = concat_images_data(image_data_a, image_data_b, how="horizontally")

    assert concatenated.image.shape == (10, 20, 4)
    np.testing.assert_array_equal(concatenated.keypoints, np.array([[1, 1], [12, 2]]))
    assert concatenated.bboxes_data[0].coords == (0, 0, 10, 10)
    assert concatenated.bboxes_data[1].coords == (10, 0, 20, 10)
    assert concatenated.bboxes_data[0].additional_bboxes_data[0].coords == (1, 1, 4, 4)
    assert concatenated.bboxes_data[1].additional_bboxes_data[0].coords == (12, 2, 15, 5)


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


def test_non_max_suppression_respects_iou_threshold_for_partial_overlaps():
    image_data = ImageData(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        bboxes_data=[
            BboxData(xmin=0, ymin=0, xmax=10, ymax=10, detection_score=0.9, label="a"),
            BboxData(xmin=8, ymin=8, xmax=18, ymax=18, detection_score=0.8, label="b"),
        ],
    )

    result = non_max_suppression_image_data(image_data, iou=0.5)

    assert [bbox_data.label for bbox_data in result.bboxes_data] == ["a", "b"]


def test_split_by_grid_and_recursive_bbox_collection():
    child = BboxData(xmin=1, ymin=1, xmax=2, ymax=2, label="child")
    parent = BboxData(xmin=0, ymin=0, xmax=4, ymax=4, label="parent", additional_bboxes_data=[child])
    image_data = ImageData(image=np.zeros((8, 8, 3), dtype=np.uint8), bboxes_data=[parent])

    grid = split_by_grid(size=(8, 8), n_rows=2, n_cols=2, x_window_size=4, y_window_size=4, x_offset=0, y_offset=0)
    all_bboxes = get_all_bboxes_data_in_image_data(image_data)

    assert [bbox.coords for bbox in grid] == [(0, 0, 4, 4), (4, 0, 8, 4), (0, 4, 4, 8), (4, 4, 8, 8)]
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


def test_image_data_transforms_drop_source_image_path_when_image_is_materialized():
    image_data = ImageData(
        image=np.zeros((20, 30, 3), dtype=np.uint8),
        image_path="/source/image.png",
        bboxes_data=[
            BboxData(
                xmin=5,
                ymin=5,
                xmax=15,
                ymax=15,
                additional_bboxes_data=[BboxData(xmin=7, ymin=7, xmax=10, ymax=10)],
            )
        ],
    )

    transformed_images_data = [
        resize_image_data(image_data, size=(15, 10)),
        thumbnail_image_data(image_data, size=10),
        crop_image_data(
            image_data,
            xmin=0,
            ymin=0,
            xmax=20,
            ymax=15,
            allow_negative_and_large_coords=False,
            remove_bad_coords=False,
        ),
        rotate_image_data(image_data, angle=90),
        apply_perspective_transform_to_image_data(
            image_data,
            perspective_matrix=np.eye(3, dtype=np.float32),
            result_width=30,
            result_height=20,
            allow_negative_and_large_coords=False,
            remove_bad_coords=False,
        ),
    ]

    for transformed in transformed_images_data:
        assert transformed.image is not None
        assert transformed.image_path is None
        assert all(bbox_data.image_path is None for bbox_data in _all_bboxes(transformed.bboxes_data))
        assert all(bbox_data.image is transformed.image for bbox_data in _all_bboxes(transformed.bboxes_data))


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
