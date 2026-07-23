import numpy as np
import pytest

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.metrics.keypoints import get_df_keypoints_metrics

pytest.importorskip("ultralytics")


def _image_with_keypoints(keypoints, detection_score=0.9, label="person"):
    return ImageData(
        bboxes_data=[
            BboxData(
                xmin=0,
                ymin=0,
                xmax=40,
                ymax=40,
                label=label,
                keypoints=keypoints,
                detection_score=detection_score,
            )
        ]
    )


def test_get_df_keypoints_metrics_perfect_match():
    keypoints = np.array([[10, 10], [20, 10], [15, 20], [10, 30], [20, 30]], dtype=int)
    true_images_data = [_image_with_keypoints(keypoints)]
    pred_images_data = [_image_with_keypoints(keypoints, detection_score=0.95)]

    df = get_df_keypoints_metrics(true_images_data, pred_images_data, class_names=["person"])

    assert df.loc["images_support", "value"] == 1
    assert df.loc["support", "value"] == 1
    assert df.loc["pose_P", "value"] == pytest.approx(1.0)
    assert df.loc["pose_R", "value"] == pytest.approx(1.0)
    assert df.loc["pose_mAP50", "value"] == pytest.approx(0.995, abs=0.01)
    assert df.loc["pose_mAP50_95", "value"] == pytest.approx(0.995, abs=0.01)


def test_get_df_keypoints_metrics_mismatch():
    keypoints = np.array([[10, 10], [20, 10], [15, 20], [10, 30], [20, 30]], dtype=int)
    true_images_data = [_image_with_keypoints(keypoints)]
    pred_images_data = [_image_with_keypoints(keypoints + 50)]

    df = get_df_keypoints_metrics(true_images_data, pred_images_data, class_names=["person"])

    assert df.loc["pose_P", "value"] == pytest.approx(0.0)
    assert df.loc["pose_R", "value"] == pytest.approx(0.0)
    assert df.loc["pose_mAP50", "value"] == pytest.approx(0.0)
    assert df.loc["pose_mAP50_95", "value"] == pytest.approx(0.0)


def test_get_df_keypoints_metrics_empty():
    df = get_df_keypoints_metrics([ImageData(bboxes_data=[])], [ImageData(bboxes_data=[])])

    assert df.loc["images_support", "value"] == 1
    assert df.loc["support", "value"] == 0
    assert df.loc["pose_mAP50", "value"] == pytest.approx(0.0)
