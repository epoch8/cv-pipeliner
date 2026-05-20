import json

import numpy as np
import pytest
from PIL import Image

from cv_pipeliner.core.data import BboxData, ImageData


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
