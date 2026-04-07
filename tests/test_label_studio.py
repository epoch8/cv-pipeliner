from cv_pipeliner.utils.label_studio import convert_annotation_to_image_data


KEYPOINT_LABELS = ["nose", "left_eye"]
MASK_LABELS = ["person"]


def _bbox_result(result_id: str, x: float, y: float, width: float, height: float):
    return {
        "id": result_id,
        "type": "rectanglelabels",
        "from_name": "bbox",
        "to_name": "image",
        "original_width": 100,
        "original_height": 100,
        "image_rotation": 0,
        "value": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rotation": 0,
            "rectanglelabels": ["person"],
        },
    }


def _keypoint_result(result_id: str, x: float, y: float, label: str):
    return {
        "id": result_id,
        "type": "keypointlabels",
        "from_name": "kp",
        "to_name": "image",
        "original_width": 100,
        "original_height": 100,
        "image_rotation": 0,
        "value": {"x": x, "y": y, "width": 0.8, "keypointlabels": [label]},
    }


def _mask_result(result_id: str, points):
    return {
        "id": result_id,
        "type": "polygonlabels",
        "from_name": "mask",
        "to_name": "image",
        "original_width": 100,
        "original_height": 100,
        "image_rotation": 0,
        "value": {"points": points, "polygonlabels": ["person"]},
    }


def _relation_result(from_id: str, to_id: str):
    return {"type": "relation", "from_id": from_id, "to_id": to_id, "direction": "bi"}


def test_related_keypoints_and_masks_are_attached_to_bbox():
    annotation = {
        "result": [
            _bbox_result("bbox0", 10, 10, 20, 20),
            _keypoint_result("kp0", 12, 12, "nose"),
            _keypoint_result("kp1", 18, 12, "left_eye"),
            _mask_result("mask0", [[10, 10], [30, 10], [30, 30], [10, 30]]),
            _relation_result("kp0", "bbox0"),
            _relation_result("kp1", "bbox0"),
            _relation_result("mask0", "bbox0"),
        ]
    }

    image_data = convert_annotation_to_image_data(
        annotation=annotation,
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=KEYPOINT_LABELS,
        mask_from_name="mask",
        mask_labels=MASK_LABELS,
    )

    assert len(image_data.bboxes_data) == 1
    bbox = image_data.bboxes_data[0]
    assert bbox.keypoints is not None
    assert bbox.keypoints.tolist() == [[12, 12], [18, 12]]
    assert bbox.mask is not None
    assert len(bbox.mask) == 1
    assert image_data.keypoints.tolist() == []
    assert image_data.mask == []


def test_relations_are_order_independent():
    annotation = {
        "result": [
            _relation_result("kp0", "bbox0"),
            _relation_result("mask0", "bbox0"),
            _bbox_result("bbox0", 10, 10, 20, 20),
            _keypoint_result("kp0", 12, 12, "nose"),
            _mask_result("mask0", [[10, 10], [30, 10], [30, 30], [10, 30]]),
        ]
    }

    image_data = convert_annotation_to_image_data(
        annotation=annotation,
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=KEYPOINT_LABELS,
        mask_from_name="mask",
        mask_labels=MASK_LABELS,
    )

    bbox = image_data.bboxes_data[0]
    assert bbox.keypoints is not None
    assert bbox.keypoints.tolist() == [[12, 12]]
    assert bbox.mask is not None
    assert len(bbox.mask) == 1
    assert image_data.keypoints.tolist() == []
    assert image_data.mask == []


def test_mixed_related_and_unrelated_items():
    annotation = {
        "result": [
            _bbox_result("bbox0", 10, 10, 20, 20),
            _keypoint_result("kp0", 12, 12, "nose"),
            _keypoint_result("kp1", 50, 50, "left_eye"),
            _mask_result("mask0", [[10, 10], [30, 10], [30, 30], [10, 30]]),
            _mask_result("mask1", [[40, 40], [60, 40], [60, 60], [40, 60]]),
            _relation_result("kp0", "bbox0"),
            _relation_result("mask0", "bbox0"),
        ]
    }

    image_data = convert_annotation_to_image_data(
        annotation=annotation,
        bboxes_from_name="bbox",
        keypoints_from_name="kp",
        keypoints_labels=KEYPOINT_LABELS,
        mask_from_name="mask",
        mask_labels=MASK_LABELS,
    )

    bbox = image_data.bboxes_data[0]
    assert bbox.keypoints is not None
    assert bbox.keypoints.tolist() == [[12, 12]]
    assert bbox.mask is not None
    assert len(bbox.mask) == 1
    assert image_data.keypoints.tolist() == [[50, 50]]
    assert len(image_data.mask) == 1

