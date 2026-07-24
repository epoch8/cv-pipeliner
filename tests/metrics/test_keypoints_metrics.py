import tempfile
from pathlib import Path

import dataframe_image as dfi
import imageio
import numpy as np
import pytest
from PIL import Image
from test_utils import visualize_images_data_with_overlay

from cv_pipeliner.core.data import BboxData, ImageData, KeypointVisibility
from cv_pipeliner.metrics.keypoints import get_df_keypoints_metrics
from cv_pipeliner.utils.images import concat_images

pytest.importorskip("ultralytics")

test_dir = Path(__file__).parent / "images"
test_dir.mkdir(exist_ok=True, parents=True)
image_path = Path(__file__).parent / "original.jpg"

CLASS_NAMES = ["person", "dog"]


def _keypoints(x: int, y: int, dx: int = 0, dy: int = 0) -> np.ndarray:
    """5 keypoints arranged as a small stick figure relative to (x, y)."""
    return np.array(
        [
            [x + dx, y + dy],
            [x + 40 + dx, y + dy],
            [x + 20 + dx, y + 40 + dy],
            [x + dx, y + 80 + dy],
            [x + 40 + dx, y + 80 + dy],
        ],
        dtype=int,
    )


# Left person — perfect OKS match with pred_person_left
true_person_left = BboxData(
    image_path=image_path,
    xmin=200,
    ymin=200,
    xmax=400,
    ymax=500,
    label="person",
    keypoints=_keypoints(240, 240),
)
pred_person_left = BboxData(  # <-> true_person_left; pose TP
    image_path=image_path,
    xmin=205,
    ymin=205,
    xmax=395,
    ymax=495,
    label="person",
    keypoints=_keypoints(240, 240),
    detection_score=0.95,
)

# Right person — bbox overlaps but keypoints are far → pose mismatch
true_person_right = BboxData(
    image_path=image_path,
    xmin=1200,
    ymin=200,
    xmax=1500,
    ymax=500,
    label="person",
    keypoints=_keypoints(1280, 240),
)
pred_person_right = BboxData(  # <-> true_person_right spatially, but keypoints far; pose FN/FP
    image_path=image_path,
    xmin=1210,
    ymin=210,
    xmax=1490,
    ymax=490,
    label="person",
    keypoints=_keypoints(1280, 240, dx=200, dy=200),
    detection_score=0.9,
)

# Dog — GT dog predicted as person (wrong class): not a pose TP for dog/person
true_dog = BboxData(
    image_path=image_path,
    xmin=700,
    ymin=700,
    xmax=1000,
    ymax=1000,
    label="dog",
    keypoints=_keypoints(760, 760),
)
pred_dog_as_person = BboxData(  # wrong class; pose mismatch for dog class
    image_path=image_path,
    xmin=710,
    ymin=710,
    xmax=990,
    ymax=990,
    label="person",
    keypoints=_keypoints(760, 760),
    detection_score=0.85,
)

# Extra prediction with no GT pair
pred_extra_person = BboxData(  # pose FP
    image_path=image_path,
    xmin=1600,
    ymin=200,
    xmax=1800,
    ymax=400,
    label="person",
    keypoints=_keypoints(1640, 240),
    detection_score=0.7,
)

# GT bbox without keypoints — excluded from keypoints support
true_person_no_keypoints = BboxData(
    image_path=image_path,
    xmin=100,
    ymin=900,
    xmax=250,
    ymax=1100,
    label="person",
    keypoints=[],
)

true_image_data = ImageData(
    image_path=image_path,
    bboxes_data=[
        true_person_left,
        true_person_right,
        true_dog,
        true_person_no_keypoints,
    ],
)
pred_image_data = ImageData(
    image_path=image_path,
    bboxes_data=[
        pred_person_left,
        pred_person_right,
        pred_dog_as_person,
        pred_extra_person,
    ],
)

true_image_data_perfect = ImageData(
    image_path=image_path,
    bboxes_data=[true_person_left, true_person_right, true_dog],
)
pred_image_data_perfect = ImageData(
    image_path=image_path,
    bboxes_data=[
        BboxData(
            image_path=image_path,
            xmin=true_person_left.xmin,
            ymin=true_person_left.ymin,
            xmax=true_person_left.xmax,
            ymax=true_person_left.ymax,
            label="person",
            keypoints=true_person_left.keypoints,
            detection_score=0.99,
        ),
        BboxData(
            image_path=image_path,
            xmin=true_person_right.xmin,
            ymin=true_person_right.ymin,
            xmax=true_person_right.xmax,
            ymax=true_person_right.ymax,
            label="person",
            keypoints=true_person_right.keypoints,
            detection_score=0.98,
        ),
        BboxData(
            image_path=image_path,
            xmin=true_dog.xmin,
            ymin=true_dog.ymin,
            xmax=true_dog.xmax,
            ymax=true_dog.ymax,
            label="dog",
            keypoints=true_dog.keypoints,
            detection_score=0.97,
        ),
    ],
)


def _export_metrics_visual(true_image: ImageData, pred_image: ImageData, df, filename: str):
    image = visualize_images_data_with_overlay(
        image_data1=true_image, image_data2=pred_image, include_keypoints=True
    )
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        dfi.export(obj=df, fontsize=10, filename=f.name, table_conversion="matplotlib")
        df_image = imageio.v3.imread(f.name)
    total_image = concat_images(image_a=image, image_b=df_image, how="vertically", mode="RGB")
    Image.fromarray(total_image).save(test_dir / filename)


def test_get_df_keypoints_metrics_perfect_match():
    df = get_df_keypoints_metrics(
        true_images_data=[true_image_data_perfect],
        pred_images_data=[pred_image_data_perfect],
        class_names=CLASS_NAMES,
    )

    assert df.loc["images_support", "value"] == 1
    assert df.loc["support", "value"] == 3  # person, person, dog (no empty-kpts bbox)
    assert df.loc["pose_P", "value"] == pytest.approx(1.0)
    assert df.loc["pose_R", "value"] == pytest.approx(1.0)
    assert df.loc["pose_mAP50", "value"] == pytest.approx(0.995, abs=0.01)
    assert df.loc["pose_mAP50_95", "value"] == pytest.approx(0.995, abs=0.01)

    _export_metrics_visual(
        true_image_data_perfect,
        pred_image_data_perfect,
        df,
        "df_keypoints_metrics_perfect_visualized.png",
    )


def test_get_df_keypoints_metrics_mixed_matches():
    """
    Expected:
      - true_person_left ↔ pred_person_left: pose TP
      - true_person_right ↔ pred_person_right: keypoints far → no OKS match
      - true_dog ↔ pred_dog_as_person: wrong class → no match
      - pred_extra_person: FP
      - true_person_no_keypoints: excluded from support
      → support == 3 (left person, right person, dog)
    """
    df = get_df_keypoints_metrics(
        true_images_data=[true_image_data],
        pred_images_data=[pred_image_data],
        class_names=CLASS_NAMES,
    )

    assert df.loc["images_support", "value"] == 1
    assert df.loc["support", "value"] == 3
    # One good match out of mixed preds → metrics well below perfect, above zero
    assert 0.0 < float(df.loc["pose_P", "value"]) < 0.6
    assert 0.0 < float(df.loc["pose_R", "value"]) < 0.6
    assert 0.0 < float(df.loc["pose_mAP50", "value"]) < 0.6
    assert float(df.loc["pose_mAP50_95", "value"]) <= float(df.loc["pose_mAP50", "value"]) + 1e-6

    _export_metrics_visual(
        true_image_data,
        pred_image_data,
        df,
        "df_keypoints_metrics_mixed_visualized.png",
    )


def test_get_df_keypoints_metrics_complete_mismatch():
    keypoints = _keypoints(100, 100)
    true_images = [
        ImageData(
            image_path=image_path,
            bboxes_data=[
                BboxData(
                    image_path=image_path,
                    xmin=50,
                    ymin=50,
                    xmax=200,
                    ymax=200,
                    label="person",
                    keypoints=keypoints,
                )
            ],
        )
    ]
    pred_images = [
        ImageData(
            image_path=image_path,
            bboxes_data=[
                BboxData(
                    image_path=image_path,
                    xmin=50,
                    ymin=50,
                    xmax=200,
                    ymax=200,
                    label="person",
                    keypoints=keypoints + 300,
                    detection_score=0.9,
                )
            ],
        )
    ]

    df = get_df_keypoints_metrics(true_images, pred_images, class_names=["person"])

    assert df.loc["support", "value"] == 1
    assert df.loc["pose_P", "value"] == pytest.approx(0.0)
    assert df.loc["pose_R", "value"] == pytest.approx(0.0)
    assert df.loc["pose_mAP50", "value"] == pytest.approx(0.0)
    assert df.loc["pose_mAP50_95", "value"] == pytest.approx(0.0)


def test_get_df_keypoints_metrics_empty():
    df = get_df_keypoints_metrics(
        [ImageData(image_path=image_path, bboxes_data=[])],
        [ImageData(image_path=image_path, bboxes_data=[])],
    )

    assert df.loc["images_support", "value"] == 1
    assert df.loc["support", "value"] == 0
    assert df.loc["pose_P", "value"] == pytest.approx(0.0)
    assert df.loc["pose_R", "value"] == pytest.approx(0.0)
    assert df.loc["pose_mAP50", "value"] == pytest.approx(0.0)
    assert df.loc["pose_mAP50_95", "value"] == pytest.approx(0.0)


def test_get_df_keypoints_metrics_skips_empty_keypoints_in_support():
    true_images = [
        ImageData(
            image_path=image_path,
            bboxes_data=[
                true_person_no_keypoints,
                true_person_left,
            ],
        )
    ]
    pred_images = [
        ImageData(
            image_path=image_path,
            bboxes_data=[pred_person_left],
        )
    ]

    df = get_df_keypoints_metrics(true_images, pred_images, class_names=CLASS_NAMES)

    assert df.loc["support", "value"] == 1
    assert df.loc["pose_P", "value"] == pytest.approx(1.0)
    assert df.loc["pose_R", "value"] == pytest.approx(1.0)


def test_get_df_keypoints_metrics_multi_image_aggregation():
    df_single = get_df_keypoints_metrics(
        [true_image_data_perfect],
        [pred_image_data_perfect],
        class_names=CLASS_NAMES,
    )
    df_multi = get_df_keypoints_metrics(
        [true_image_data_perfect, true_image_data_perfect],
        [pred_image_data_perfect, pred_image_data_perfect],
        class_names=CLASS_NAMES,
    )

    assert df_multi.loc["images_support", "value"] == 2
    assert df_multi.loc["support", "value"] == 2 * df_single.loc["support", "value"]
    assert df_multi.loc["pose_P", "value"] == pytest.approx(df_single.loc["pose_P", "value"])
    assert df_multi.loc["pose_R", "value"] == pytest.approx(df_single.loc["pose_R", "value"])
    assert df_multi.loc["pose_mAP50", "value"] == pytest.approx(df_single.loc["pose_mAP50", "value"], abs=0.01)


def test_get_df_keypoints_metrics_without_class_names():
    df = get_df_keypoints_metrics(
        [true_image_data_perfect],
        [pred_image_data_perfect],
        class_names=None,
    )
    # All labels collapse to class 0 → still perfect pose match
    assert df.loc["support", "value"] == 3
    assert df.loc["pose_P", "value"] == pytest.approx(1.0)
    assert df.loc["pose_mAP50", "value"] == pytest.approx(0.995, abs=0.01)


def test_get_df_keypoints_metrics_visibility_mismatch_raises():
    bad = BboxData(
        image_path=image_path,
        xmin=0,
        ymin=0,
        xmax=100,
        ymax=100,
        label="person",
        keypoints=_keypoints(10, 10),
        keypoints_visibility=[KeypointVisibility.LABELED_AND_VISIBLE] * 3,  # 3 != 5
    )
    with pytest.raises(ValueError, match="keypoints_visibility length"):
        get_df_keypoints_metrics(
            [ImageData(image_path=image_path, bboxes_data=[bad])],
            [ImageData(image_path=image_path, bboxes_data=[pred_person_left])],
            class_names=["person"],
        )


def test_get_df_keypoints_metrics_sigma_length_mismatch_raises():
    with pytest.raises(ValueError, match="sigma length"):
        get_df_keypoints_metrics(
            [true_image_data_perfect],
            [pred_image_data_perfect],
            class_names=CLASS_NAMES,
            sigma=np.ones(3, dtype=np.float32),  # nkpt == 5
        )


def test_get_df_keypoints_metrics_custom_sigma_perfect_match():
    sigma = np.ones(5, dtype=np.float32) / 5.0
    df = get_df_keypoints_metrics(
        [true_image_data_perfect],
        [pred_image_data_perfect],
        class_names=CLASS_NAMES,
        sigma=sigma,
    )
    assert df.loc["pose_P", "value"] == pytest.approx(1.0)
    assert df.loc["pose_mAP50", "value"] == pytest.approx(0.995, abs=0.01)
