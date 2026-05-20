import base64
import io

import numpy as np
import pytest
from PIL import Image

from cv_pipeliner.utils.images import (
    concat_images,
    denormalize_bboxes,
    get_image_b64,
    get_thumbnail_resize,
    open_image,
    rescale_bboxes_with_pad,
    thumbnail_image,
)


def _png_bytes(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def test_open_image_reads_bytes_base64_pil_grayscale_and_rgba():
    gray = np.zeros((3, 4), dtype=np.uint8)
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 255

    gray_rgb = open_image(_png_bytes(gray), open_as_rgb=True)
    rgba_rgb = open_image(Image.fromarray(rgba), open_as_rgb=True)
    b64_rgb = open_image(base64.b64encode(_png_bytes(gray)).decode(), open_as_rgb=True)

    assert gray_rgb.shape == (3, 4, 3)
    assert rgba_rgb.shape == (3, 4, 3)
    assert b64_rgb.shape == (3, 4, 3)


def test_concat_images_horizontal_vertical_modes_and_invalid_values():
    image_a = np.zeros((4, 5, 3), dtype=np.uint8)
    image_b = np.zeros((6, 3, 3), dtype=np.uint8)

    horizontal = concat_images(image_a, image_b, how="horizontally", mode="RGB", between_edge_width=2)
    vertical = concat_images(image_a, image_b, how="vertically", mode="L", between_edge_width=1)

    assert horizontal.shape == (6, 10, 3)
    assert vertical.shape == (11, 5)
    with pytest.raises(ValueError):
        concat_images(image_a, image_b, how="diagonal")
    with pytest.raises(ValueError):
        concat_images(image_a, image_b, mode="BAD")


def test_thumbnail_helpers_preserve_aspect_ratio():
    assert get_thumbnail_resize((100, 50), (20, 20)) == (20, 10)
    thumbnail = thumbnail_image(np.zeros((50, 100, 3), dtype=np.uint8), size=(20, 20))

    assert thumbnail.shape[:2] == (10, 20)


def test_bbox_coordinate_helpers():
    denormalized = denormalize_bboxes([(0.1, 0.2, 0.9, 1.2)], image_width=10, image_height=20)
    rescaled = rescale_bboxes_with_pad([(2, 4, 8, 12)], current_width=20, current_height=20, target_width=10, target_height=10)

    assert denormalized.tolist() == [[1, 4, 9, 19]]
    assert rescaled.tolist() == [[1, 2, 4, 6]]


def test_get_image_b64_returns_decodable_image():
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    decoded = open_image(get_image_b64(image, "PNG"), open_as_rgb=True)

    assert decoded.shape == image.shape
