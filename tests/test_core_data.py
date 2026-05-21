import json
import sys

import numpy as np
import pytest
from PIL import Image

from cv_pipeliner.core.data import BboxData, ImageData


def _all_bboxes(bboxes_data):
    result = []
    for bbox_data in bboxes_data:
        result.append(bbox_data)
        result.extend(_all_bboxes(bbox_data.additional_bboxes_data))
    return result


def test_image_data_from_array_sets_meta_and_normalizes_keypoints():
    image_data = ImageData(image=np.zeros((10, 20, 3), dtype=np.uint8), keypoints=[(10, 5)])

    assert image_data.get_image_size() == (20, 10)
    np.testing.assert_allclose(image_data.keypoints_n, np.array([[0.5, 0.5]]))


def test_image_data_open_image_from_path_and_json_roundtrip(tmp_dir):
    image_path = tmp_dir / "image.png"
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(image_path)
    image_data = ImageData(image_path=image_path, label="sample", bboxes_data=[BboxData(xmin=1, ymin=1, xmax=3, ymax=3)])

    opened = image_data.open_image(inplace=True)
    restored = ImageData.from_json(json.loads(image_data.json(force_include_meta=True)))

    assert opened.shape == (4, 5, 3)
    assert restored.image_path == image_data.image_path
    assert restored.label == "sample"
    assert restored.bboxes_data[0].coords == (1, 1, 3, 3)


def test_image_data_mask_array_is_converted_to_polygons_and_opened():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 255
    image_data = ImageData(image=np.zeros((5, 6, 3), dtype=np.uint8), mask=mask)

    opened_mask = image_data.open_mask()

    assert len(image_data.mask) >= 1
    assert opened_mask.shape == (5, 6)
    assert opened_mask.max() == 255


def test_bbox_data_coords_area_normalization_and_crop_as_image_data():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    bbox_data = BboxData(
        image=image,
        xmin=2,
        ymin=3,
        xmax=8,
        ymax=9,
        keypoints=[(4, 5)],
        mask=[[(2, 3), (8, 3), (8, 9), (2, 9)]],
        additional_bboxes_data=[BboxData(xmin=3, ymin=4, xmax=6, ymax=7)],
    )

    crop_image_data = bbox_data.open_cropped_image(return_as_image_data=True)

    assert bbox_data.coords == (2, 3, 8, 9)
    assert bbox_data.area == 49
    np.testing.assert_allclose(bbox_data.coords_n, (0.1, 0.3, 0.4, 0.9))
    assert crop_image_data.image.shape == (6, 6, 3)
    assert crop_image_data.bboxes_data[0].coords == (1, 1, 4, 4)
    np.testing.assert_array_equal(crop_image_data.keypoints, np.array([[2, 2]]))


def test_image_data_open_image_raises_when_empty():
    with pytest.raises(ValueError):
        ImageData().open_image()


def test_image_data_propagates_source_fields_to_nested_bboxes():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    child_bbox = BboxData(xmin=2, ymin=2, xmax=4, ymax=4)
    parent_bbox = BboxData(xmin=1, ymin=1, xmax=5, ymax=5, additional_bboxes_data=[child_bbox])
    image_data = ImageData(image=image, image_path="source.jpg", bboxes_data=[parent_bbox])

    assert image_data.bboxes_data[0].image_path == image_data.image_path
    assert image_data.bboxes_data[0].additional_bboxes_data[0].image_path == image_data.image_path
    assert image_data.bboxes_data[0].image is image_data.image
    assert image_data.bboxes_data[0].additional_bboxes_data[0].image is image_data.image
    assert image_data.bboxes_data[0].meta_width == 8
    assert image_data.bboxes_data[0].additional_bboxes_data[0].meta_height == 6


def test_source_field_reassignment_propagates_recursively():
    parent_bbox = BboxData(
        xmin=1,
        ymin=1,
        xmax=5,
        ymax=5,
        additional_bboxes_data=[BboxData(xmin=2, ymin=2, xmax=4, ymax=4)],
    )
    image_data = ImageData(image_path="first.jpg", meta_width=10, meta_height=12, bboxes_data=[parent_bbox])

    image_data.image_path = "second.jpg"
    image_data.meta_width = 20
    image_data.image = np.zeros((3, 4, 3), dtype=np.uint8)

    nested_bbox = image_data.bboxes_data[0].additional_bboxes_data[0]
    assert str(image_data.bboxes_data[0].image_path) == "second.jpg"
    assert str(nested_bbox.image_path) == "second.jpg"
    assert image_data.bboxes_data[0].image is image_data.image
    assert nested_bbox.image is image_data.image
    assert image_data.bboxes_data[0].meta_width == 4
    assert nested_bbox.meta_height == 3


def test_bbox_data_propagates_source_fields_to_new_additional_bboxes():
    bbox_data = BboxData(image_path="source.jpg", xmin=1, ymin=1, xmax=5, ymax=5, meta_width=10, meta_height=12)

    bbox_data.additional_bboxes_data = [BboxData(xmin=2, ymin=2, xmax=4, ymax=4)]

    assert str(bbox_data.additional_bboxes_data[0].image_path) == "source.jpg"
    assert bbox_data.additional_bboxes_data[0].meta_width == 10


def test_image_data_image_propagation_keeps_single_array_reference_for_many_bboxes():
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    bboxes_data = [
        BboxData(
            xmin=0,
            ymin=0,
            xmax=10,
            ymax=10,
            additional_bboxes_data=[BboxData(xmin=1, ymin=1, xmax=5, ymax=5) for _ in range(3)],
        )
        for _ in range(500)
    ]
    refcount_before = sys.getrefcount(image)

    image_data = ImageData(image=image, bboxes_data=bboxes_data)
    all_bboxes = _all_bboxes(image_data.bboxes_data)

    assert len(all_bboxes) == 2000
    assert {id(bbox_data.image) for bbox_data in all_bboxes} == {id(image_data.image)}
    assert all(bbox_data.image is image for bbox_data in all_bboxes)
    assert sys.getrefcount(image) == refcount_before + len(all_bboxes) + 1

    new_image = np.ones((128, 256, 3), dtype=np.uint8)
    new_refcount_before = sys.getrefcount(new_image)
    image_data.image = new_image

    assert {id(bbox_data.image) for bbox_data in all_bboxes} == {id(new_image)}
    assert all(bbox_data.image is new_image for bbox_data in all_bboxes)
    assert all((bbox_data.meta_width, bbox_data.meta_height) == (256, 128) for bbox_data in all_bboxes)
    assert sys.getrefcount(new_image) == new_refcount_before + len(all_bboxes) + 1


def test_bbox_data_image_propagation_keeps_single_array_reference_for_many_additional_bboxes():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    bbox_data = BboxData(
        image=image,
        xmin=0,
        ymin=0,
        xmax=20,
        ymax=20,
        additional_bboxes_data=[
            BboxData(
                xmin=1,
                ymin=1,
                xmax=10,
                ymax=10,
                additional_bboxes_data=[BboxData(xmin=2, ymin=2, xmax=5, ymax=5)],
            )
            for _ in range(500)
        ],
    )
    all_children = _all_bboxes(bbox_data.additional_bboxes_data)

    assert len(all_children) == 1000
    assert {id(child.image) for child in all_children} == {id(image)}
    assert all(child.image is image for child in all_children)

    new_image = np.ones((64, 96, 3), dtype=np.uint8)
    bbox_data.image = new_image

    assert {id(child.image) for child in all_children} == {id(new_image)}
    assert all(child.image is new_image for child in all_children)
    assert all((child.meta_width, child.meta_height) == (96, 64) for child in all_children)


def test_image_path_reassignment_propagates_to_many_bboxes_without_touching_image_reference():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image_data = ImageData(
        image=image,
        image_path="first.jpg",
        bboxes_data=[
            BboxData(xmin=0, ymin=0, xmax=10, ymax=10, additional_bboxes_data=[BboxData(xmin=1, ymin=1, xmax=5, ymax=5)])
            for _ in range(300)
        ],
    )
    all_bboxes = _all_bboxes(image_data.bboxes_data)

    image_data.image_path = "second.jpg"

    assert all(str(bbox_data.image_path) == "second.jpg" for bbox_data in all_bboxes)
    assert all(bbox_data.image is image for bbox_data in all_bboxes)
    assert all((bbox_data.meta_width, bbox_data.meta_height) == (32, 32) for bbox_data in all_bboxes)


def test_meta_reassignment_without_image_propagates_to_many_bboxes():
    image_data = ImageData(
        image_path="first.jpg",
        meta_width=10,
        meta_height=20,
        bboxes_data=[
            BboxData(xmin=0, ymin=0, xmax=10, ymax=10, additional_bboxes_data=[BboxData(xmin=1, ymin=1, xmax=5, ymax=5)])
            for _ in range(300)
        ],
    )
    all_bboxes = _all_bboxes(image_data.bboxes_data)

    image_data.meta_width = 100
    image_data.meta_height = 200

    assert all(bbox_data.image is None for bbox_data in all_bboxes)
    assert all((bbox_data.meta_width, bbox_data.meta_height) == (100, 200) for bbox_data in all_bboxes)
