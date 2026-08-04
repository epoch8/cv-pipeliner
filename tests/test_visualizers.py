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


def test_visualize_image_data_respects_keypoints_visibility_and_scores():
    from cv_pipeliner.core.data import KeypointVisibility

    blank = np.zeros((40, 40, 3), dtype=np.uint8)
    image_data = ImageData(
        image=blank.copy(),
        keypoints=[(5, 5), (20, 20), (35, 35)],
        keypoints_visibility=[
            KeypointVisibility.NOT_LABELED,
            KeypointVisibility.LABELED_NOT_VISIBLE,
            KeypointVisibility.LABELED_AND_VISIBLE,
        ],
        keypoints_scores=[0.1, 0.5, 0.9],
        keypoints_labels=["a", "b", "c"],
        bboxes_data=[
            BboxData(
                image=blank.copy(),
                xmin=1,
                ymin=1,
                xmax=30,
                ymax=30,
                label="person",
                keypoints=[(8, 8), (15, 15), (25, 25)],
                keypoints_visibility=[0, 1, 2],
                keypoints_scores=[0.2, 0.6, 0.95],
                keypoints_labels=["nose", "left_eye", "right_eye"],
            )
        ],
    )

    without_scores = visualize_image_data(
        image_data,
        include_keypoints=True,
        include_keypoint_scores=False,
        thickness=1,
        fontsize=8,
        keypoints_radius=2,
    )
    with_scores = visualize_image_data(
        image_data,
        include_keypoints=True,
        include_keypoint_scores=True,
        thickness=1,
        fontsize=8,
        keypoints_radius=2,
    )
    with_labels = visualize_image_data(
        image_data,
        include_keypoints=True,
        include_keypoints_labels=True,
        keypoints_fontsize=10,
        thickness=1,
        fontsize=8,
        keypoints_radius=2,
    )

    assert without_scores.shape == (40, 40, 3)
    assert with_scores.shape == (40, 40, 3)
    assert with_labels.shape == (40, 40, 3)
    assert without_scores.sum() > 0
    assert with_scores.sum() > without_scores.sum()
    assert with_labels.sum() > without_scores.sum()
    # NOT_LABELED image-level keypoint at (5,5) should stay blank neighborhood
    assert blank[5, 5].sum() == 0
    assert without_scores[5, 5].sum() == 0

    without_visibility = visualize_image_data(
        image_data,
        include_keypoints=True,
        include_keypoints_visibility=False,
        thickness=1,
        fontsize=8,
        keypoints_radius=2,
    )
    # With visibility ignored, NOT_LABELED point at (5,5) is drawn
    assert without_visibility[5, 5].sum() > 0


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


def test_place_keypoint_caption_box_flips_below_near_top():
    from cv_pipeliner.visualizers.core.image_data import _place_keypoint_caption_box

    left, top, right, bottom, text_bottom = _place_keypoint_caption_box(
        x=10,
        y=5,
        keypoints_radius=2,
        text_width=40,
        text_height=10,
        margin=1,
        image_width=200,
        image_height=200,
        placed_rects=[],
    )
    # Near the image top → prefer a slot below / beside, not off-canvas above.
    assert top >= 0
    assert text_bottom == bottom
    assert bottom <= 5 + 2 + 10 + 2 * 1 + 5


def test_place_keypoint_caption_box_nudges_to_avoid_overlap():
    from cv_pipeliner.visualizers.core.image_data import _place_keypoint_caption_box, _rects_overlap

    first = _place_keypoint_caption_box(
        x=20,
        y=50,
        keypoints_radius=2,
        text_width=40,
        text_height=10,
        margin=1,
        image_width=200,
        image_height=200,
        placed_rects=[],
    )
    second = _place_keypoint_caption_box(
        x=22,
        y=52,
        keypoints_radius=2,
        text_width=40,
        text_height=10,
        margin=1,
        image_width=200,
        image_height=200,
        placed_rects=[first[:4]],
    )
    assert not _rects_overlap(first[:4], second[:4])
    # Stay near the keypoint instead of stacking far away.
    assert abs(second[1] - 52) < 80


def test_place_keypoint_caption_box_avoids_keypoint_circle_and_bbox_label():
    from cv_pipeliner.visualizers.core.image_data import (
        _caption_center,
        _keypoint_circle_rect,
        _place_keypoint_caption_box,
        _rects_overlap,
    )

    x, y, radius = 30.0, 40.0, 5
    circle = _keypoint_circle_rect(x, y, radius)
    # Occupies the default right-of-point slot next to the circle.
    bbox_label = (x + radius + 1, y - 20, x + radius + 1 + 50, y - 5)
    left, top, right, bottom, _ = _place_keypoint_caption_box(
        x=x,
        y=y,
        keypoints_radius=radius,
        text_width=40,
        text_height=10,
        margin=1,
        image_width=200,
        image_height=200,
        placed_rects=[circle, bbox_label],
    )
    caption = (left, top, right, bottom)
    assert not _rects_overlap(caption, circle)
    assert not _rects_overlap(caption, bbox_label)
    cx, cy = _caption_center(caption)
    assert (cx - x) ** 2 + (cy - y) ** 2 < 120**2


def test_keypoint_caption_leader_polyline_is_l_shaped():
    from cv_pipeliner.visualizers.core.image_data import _keypoint_caption_leader_polyline

    # Caption to the right of the point → horizontal then vertical bend.
    points = _keypoint_caption_leader_polyline(10, 20, 3, (30, 10, 70, 25))
    assert len(points) >= 2
    assert points[0][0] > 10  # starts outside the circle toward the caption
    # Elbow shares keypoint y or caption attach x.
    assert any(abs(p[1] - 20) < 1e-6 for p in points) or any(abs(p[0] - 30) < 1e-6 for p in points)


def test_keypoint_caption_leader_avoids_foreign_text():
    from cv_pipeliner.visualizers.core.image_data import (
        _axis_aligned_segment_hits_rect,
        _keypoint_caption_leader_polyline,
        _polyline_text_hit_count,
    )

    # Caption sits above the bbox label; naive vertical route crosses "giraffe".
    caption = (100.0, 40.0, 190.0, 55.0)
    giraffe = (100.0, 70.0, 160.0, 90.0)
    kx, ky = 130.0, 110.0
    naive = [(kx, ky - 5), (kx, caption[3]), ((caption[0] + caption[2]) / 2, caption[3])]
    assert _polyline_text_hit_count(naive, [giraffe], pad=3.0) >= 1
    assert _axis_aligned_segment_hits_rect(kx, ky - 5, kx, caption[3], giraffe, pad=3.0)

    points = _keypoint_caption_leader_polyline(kx, ky, 5, caption, text_rects=[giraffe])
    assert _polyline_text_hit_count(points, [giraffe], pad=3.0) == 0
    # Detour should go around (more than a plain L through the label).
    assert len(points) >= 3


def test_keypoint_caption_leader_avoids_bbox_label_like_main_head():
    """Regression: image-level main_head above bbox must not cross 'giraffe' text."""
    from cv_pipeliner.visualizers.core.image_data import (
        _keypoint_caption_leader_polyline,
        _polyline_text_hit_count,
    )

    # Thumbnail-scaled geometry from the giraffe notebook viz.
    giraffe = (422.0, 52.0, 452.0, 65.0)
    caption = (441.0, 41.0, 494.0, 52.0)
    kx, ky = 434.0, 70.0
    naive = [(kx, ky - 5), (kx, (caption[1] + caption[3]) / 2.0), (caption[0], (caption[1] + caption[3]) / 2.0)]
    assert _polyline_text_hit_count(naive, [giraffe], pad=3.0) >= 1
    points = _keypoint_caption_leader_polyline(kx, ky, 5, caption, text_rects=[giraffe])
    assert _polyline_text_hit_count(points, [giraffe], pad=3.0) == 0
    assert len(points) >= 3


def test_keypoint_color_prefers_label_to_color():
    from cv_pipeliner.visualizers.core.image_data import _resolve_keypoint_color

    assert _resolve_keypoint_color(0, "nose", label_to_color={"nose": "cyan"}, default_color="red") == "cyan"
    assert _resolve_keypoint_color(0, "unknown", label_to_color={"nose": "cyan"}, default_color="red") == "red"
    assert _resolve_keypoint_color(0, None, label_to_color={"nose": "cyan"}, default_color="red") == "red"
    # Without label_to_color, keep legacy index palette even if default_color is set.
    assert _resolve_keypoint_color(1, None, label_to_color=None, default_color="red") == "Green"
    assert _resolve_keypoint_color(1, None, label_to_color=None, default_color=None) == "Green"


def test_visualize_image_data_applies_label_to_color_to_keypoints():
    blank = np.zeros((40, 40, 3), dtype=np.uint8)
    image_data = ImageData(
        image=blank.copy(),
        keypoints=[(10, 10)],
        keypoints_labels=["nose"],
        bboxes_data=[
            BboxData(
                image=blank.copy(),
                xmin=1,
                ymin=1,
                xmax=30,
                ymax=30,
                label="person",
                keypoints=[(20, 20)],
                keypoints_labels=["left_eye"],
            )
        ],
    )
    result = visualize_image_data(
        image_data,
        include_keypoints=True,
        include_keypoints_labels=True,
        label_to_color={"nose": "red", "left_eye": "lime", "person": "blue"},
        thickness=1,
        fontsize=8,
        keypoints_radius=3,
    )
    assert result.shape == (40, 40, 3)
    assert result.sum() > 0


def test_visualize_bboxes_data_returns_image_for_selected_class():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bboxes_data = [
        BboxData(image=image, xmin=1, ymin=1, xmax=6, ymax=6, label="keep"),
        BboxData(image=image, xmin=2, ymin=2, xmax=7, ymax=7, label="skip"),
    ]

    result = visualize_bboxes_data(bboxes_data, class_name="keep", visualize_size=1)

    assert result.ndim == 3
    assert result.shape[2] == 3
