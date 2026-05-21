import numpy as np
from PIL import Image

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.visualizers.core.bbox_data import visualize_bboxes_data
from cv_pipeliner.visualizers.core.image_data import visualize_image_data


def test_visualize_image_data_without_boxes_returns_array_and_pil():
    image_data = ImageData(image=np.zeros((20, 30, 3), dtype=np.uint8))

    array_result = visualize_image_data(image_data)
    pil_result = visualize_image_data(image_data, return_as_pil_image=True)

    assert array_result.shape == (20, 30, 3)
    assert isinstance(pil_result, Image.Image)


def test_visualize_image_data_prefers_loaded_image_over_image_path():
    image_data = ImageData(
        image=np.zeros((20, 30, 3), dtype=np.uint8),
        image_path="/path/that/should/not/be/read.png",
        bboxes_data=[BboxData(xmin=1, ymin=1, xmax=10, ymax=10)],
    )

    result = visualize_image_data(image_data, exif_transpose=True)

    assert result.shape == (20, 30, 3)


def test_visualize_image_data_with_labels_scores_keypoints_and_masks():
    image_data = ImageData(
        image=np.zeros((20, 30, 3), dtype=np.uint8),
        label="scene",
        mask=[[(0, 0), (10, 0), (10, 10), (0, 10)]],
        bboxes_data=[
            BboxData(
                image=np.zeros((20, 30, 3), dtype=np.uint8),
                xmin=2,
                ymin=3,
                xmax=12,
                ymax=15,
                label="object",
                detection_score=0.9,
                keypoints=[(4, 5)],
                mask=[[(2, 3), (12, 3), (12, 15), (2, 15)]],
            )
        ],
    )

    result = visualize_image_data(
        image_data,
        include_labels=True,
        score_type="detection",
        include_keypoints=True,
        include_mask=True,
        label_to_color={"object": "red", "scene": "blue"},
        thickness=1,
        fontsize=8,
    )

    assert result.shape == (20, 30, 3)
    assert result.sum() > 0


def test_visualize_image_data_includes_additional_bboxes_and_filters_labels():
    image_data = ImageData(
        image=np.zeros((20, 30, 3), dtype=np.uint8),
        bboxes_data=[
            BboxData(
                xmin=1,
                ymin=1,
                xmax=10,
                ymax=10,
                label="parent",
                additional_bboxes_data=[BboxData(xmin=3, ymin=3, xmax=8, ymax=8, label="child")],
            )
        ],
    )

    result = visualize_image_data(
        image_data,
        include_labels=True,
        include_additional_bboxes_data=True,
        filter_by_labels=["parent"],
        thickness=1,
        fontsize=8,
    )

    assert result.shape == (20, 30, 3)


def test_visualize_bboxes_data_returns_image_for_selected_class():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bboxes_data = [
        BboxData(image=image, xmin=1, ymin=1, xmax=6, ymax=6, label="keep"),
        BboxData(image=image, xmin=2, ymin=2, xmax=7, ymax=7, label="skip"),
    ]

    result = visualize_bboxes_data(bboxes_data, class_name="keep", visualize_size=1)

    assert result.ndim == 3
    assert result.shape[2] == 3
