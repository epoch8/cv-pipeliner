import numpy as np

from cv_pipeliner import BboxData, ImageData
from cv_pipeliner.utils.label_studio import convert_annotation_to_image_data, convert_image_data_to_annotation

KEYPOINTS_LABELS = ["nose", "left_eye", "right_eye"]


def _make_bbox_keypoints_annotation() -> dict:
    return {
        "result": [
            {
                "id": "bbox0",
                "type": "rectanglelabels",
                "from_name": "bbox",
                "to_name": "image",
                "original_width": 640,
                "original_height": 480,
                "value": {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 30.0,
                    "height": 40.0,
                    "rotation": 0,
                    "rectanglelabels": ["person"],
                },
            },
            {
                "id": "bbox0_kp0",
                "type": "keypointlabels",
                "from_name": "kp",
                "to_name": "image",
                "original_width": 640,
                "original_height": 480,
                "value": {"x": 15.0, "y": 25.0, "width": 0.8, "keypointlabels": ["nose"]},
            },
            {
                "id": "bbox0_kp1",
                "type": "keypointlabels",
                "from_name": "kp",
                "to_name": "image",
                "original_width": 640,
                "original_height": 480,
                "value": {"x": 16.0, "y": 24.0, "width": 0.8, "keypointlabels": ["left_eye"]},
            },
            {
                "id": "bbox0_kp2",
                "type": "keypointlabels",
                "from_name": "kp",
                "to_name": "image",
                "original_width": 640,
                "original_height": 480,
                "value": {"x": 14.0, "y": 24.0, "width": 0.8, "keypointlabels": ["right_eye"]},
            },
            {"type": "relation", "from_id": "bbox0_kp0", "to_id": "bbox0", "direction": "bi"},
            {"type": "relation", "from_id": "bbox0_kp1", "to_id": "bbox0", "direction": "bi"},
            {"type": "relation", "from_id": "bbox0_kp2", "to_id": "bbox0", "direction": "bi"},
        ]
    }


def test_convert_annotation_to_image_data_attaches_keypoints_to_bbox():
    image_data = convert_annotation_to_image_data(
        annotation=_make_bbox_keypoints_annotation(),
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=KEYPOINTS_LABELS,
    )

    assert len(image_data.bboxes_data) == 1
    bbox = image_data.bboxes_data[0]
    assert bbox.label == "person"
    assert len(bbox.keypoints) == 3
    assert np.array_equal(bbox.keypoints[0], [96, 120])
    assert np.array_equal(bbox.keypoints[1], [102, 115])
    assert np.array_equal(bbox.keypoints[2], [90, 115])
    assert bbox.keypoints_labels == ["nose", "left_eye", "right_eye"]
    assert "keypoints_labels" not in bbox.additional_info


def test_convert_annotation_to_image_data_round_trip_keypoints():
    source = ImageData(
        image_path="test.jpg",
        meta_width=640,
        meta_height=480,
        bboxes_data=[
            BboxData(
                xmin=64,
                ymin=96,
                xmax=256,
                ymax=288,
                label="person",
                keypoints=np.array([[96, 120], [102, 115], [90, 115]]),
                meta_width=640,
                meta_height=480,
            )
        ],
    )
    annotation = {"result": convert_image_data_to_annotation(
        source,
        to_name="image",
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=KEYPOINTS_LABELS,
    )}
    restored = convert_annotation_to_image_data(
        annotation=annotation,
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=KEYPOINTS_LABELS,
        image_path="test.jpg",
    )

    assert len(restored.bboxes_data) == 1
    assert np.array_equal(restored.bboxes_data[0].keypoints, source.bboxes_data[0].keypoints)

    kp_annotations = [item for item in annotation["result"] if item.get("type") == "keypointlabels"]
    assert [item["value"]["keypointlabels"][0] for item in kp_annotations] == KEYPOINTS_LABELS
    assert all(item.get("parentID") == "bbox0" for item in kp_annotations)


def test_export_uses_keypoints_labels_over_skeleton_argument():
    image_data = convert_annotation_to_image_data(
        annotation=_make_bbox_keypoints_annotation(),
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=KEYPOINTS_LABELS,
        image_path="test.jpg",
    )
    image_data.meta_width = 640
    image_data.meta_height = 480
    exported = convert_image_data_to_annotation(
        image_data,
        to_name="image",
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=["wrong", "labels", "here"],
    )
    kp_annotations = [item for item in exported if item.get("type") == "keypointlabels"]
    assert [item["value"]["keypointlabels"][0] for item in kp_annotations] == ["nose", "left_eye", "right_eye"]


def test_export_uses_keypoints_labels_field_directly():
    image_data = ImageData(
        image_path="test.jpg",
        meta_width=640,
        meta_height=480,
        bboxes_data=[
            BboxData(
                xmin=64,
                ymin=96,
                xmax=256,
                ymax=288,
                label="person",
                keypoints=np.array([[96, 120], [102, 115], [90, 115]]),
                keypoints_labels=["nose", "left_eye", "right_eye"],
                meta_width=640,
                meta_height=480,
            )
        ],
    )
    exported = convert_image_data_to_annotation(
        image_data,
        to_name="image",
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=["wrong", "labels", "here"],
    )
    kp_annotations = [item for item in exported if item.get("type") == "keypointlabels"]
    assert [item["value"]["keypointlabels"][0] for item in kp_annotations] == ["nose", "left_eye", "right_eye"]
