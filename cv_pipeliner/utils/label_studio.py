from typing import Dict, Optional, List, Any
from cv_pipeliner.core.data import BboxData, ImageData
import numpy as np

from cv_pipeliner.utils.images import rotate_point


def parse_rectangle_labels_to_bbox_data(
    rectangle_label: Dict
) -> BboxData:
    original_height = rectangle_label['original_height']
    original_width = rectangle_label['original_width']
    height = rectangle_label['value']['height']
    width = rectangle_label['value']['width']
    xmin = rectangle_label['value']['x']
    ymin = rectangle_label['value']['y']
    label = rectangle_label['value']['rectanglelabels'][0]
    xmax = xmin + width
    ymax = ymin + height
    xmin = max(0, min(original_width - 1, xmin / 100 * original_width))
    ymin = max(0, min(original_height - 1, ymin / 100 * original_height))
    xmax = max(0, min(original_width - 1, xmax / 100 * original_width))
    ymax = max(0, min(original_height - 1, ymax / 100 * original_height))
    angle = rectangle_label['value']['rotation']
    points = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    rotated_points = [rotate_point(x=x, y=y, cx=xmin, cy=ymin, angle=angle) for (x, y) in points]
    xmin = max(0, min([x for (x, y) in rotated_points]))
    ymin = max(0, min([y for (x, y) in rotated_points]))
    xmax = max([x for (x, y) in rotated_points])
    ymax = max([y for (x, y) in rotated_points])
    xmin = max(0, min(original_width-1, xmin))
    ymin = max(0, min(original_height-1, ymin))
    xmax = max(0, min(original_width-1, xmax))
    ymax = max(0, min(original_height-1, ymax))
    bbox_data = BboxData(
        xmin=round(xmin),
        ymin=round(ymin),
        xmax=round(xmax),
        ymax=round(ymax),
        label=label,
        meta_height=original_height,
        meta_width=original_width
    )
    return bbox_data


def convert_image_data_to_rectangle_labels(
    image_data: ImageData,
    from_name: str,
    to_name: str,
    keypoints_from_name: Optional[str] = None,
    without_exif_tag: bool = True
) -> Dict:
    if without_exif_tag:
        # При импорте разметки в LS не учитываются Exif теги, но при экспорте оно учитывается
        im_width, im_height = image_data.get_image_size_without_exif_tag()
    else:
        im_width, im_height = image_data.get_image_size()
    rectangle_labels = []
    for bbox_data in image_data.bboxes_data:
        rectangle_label = {
            "original_width": im_width,
            "original_height": im_height,
            "image_rotation": 0,
            "value": {
                "x": bbox_data.xmin / im_width * 100,
                "y": bbox_data.ymin / im_height * 100,
                "width": (bbox_data.xmax - bbox_data.xmin) / im_width * 100,
                "height": (bbox_data.ymax - bbox_data.ymin) / im_height * 100,
                "rotation": 0,
                "rectanglelabels": [bbox_data.label]
            },
            "from_name": from_name,
            "to_name": to_name,
            "type": "rectanglelabels"
        }
    return rectangle_labels


def parse_polygon_label_to_bbox_data(
    polygon_label: Dict
) -> BboxData:
    original_height = polygon_label['original_height']
    original_width = polygon_label['original_width']
    keypoints = []
    for (x, y) in polygon_label['value']['points']:
        x = x / 100 * polygon_label['original_width']
        y = y / 100 * polygon_label['original_height']
        keypoints.append([round(max(0, min(original_width - 1, x))), round(max(0, min(original_height - 1, y)))])
    keypoints = np.array(keypoints)
    bbox_data = BboxData(
        xmin=round(np.min(keypoints[:, 0])),
        ymin=round(np.min(keypoints[:, 1])),
        xmax=round(np.max(keypoints[:, 0])),
        ymax=np.max(keypoints[:, 1]),
        keypoints=round(keypoints),
        label=polygon_label['value']['polygonlabels'][0]
    )
    return bbox_data


def convert_image_data_to_polygon_label(
    image_data: ImageData,
    from_name: str,
    to_name: str,
    polygonlabels: str,
) -> Dict:
    im_width, im_height = image_data.get_image_size()
    rectangle_labels = []
    for bbox_data in image_data.bboxes_data:
        rectangle_labels.append({
            "original_width": im_width,
            "original_height": im_height,
            "image_rotation": 0,
            "value": {
                "points": [
                    [x * 100 / im_width, y * 100 / im_height]
                    for x, y in bbox_data.keypoints
                ],
                "polygonlabels": [polygonlabels]
            },
            "from_name": from_name,
            "to_name": to_name,
            "type": "polygonlabels"
        })
    return rectangle_labels


def convert_image_data_to_keypoint_label(
    image_data: ImageData,
    from_name: str,
    to_name: str,
    keypoints_labels: List[str],
    keypoints_width: float = 0.8
) -> Dict:
    im_width, im_height = image_data.get_image_size()
    keypoints_labels_json = []
    keypoints_to_be_added = [image_data.keypoints] + [bbox_data.keypoints for bbox_data in image_data.bboxes_data]
    for keypoints in keypoints_to_be_added:
        if len(keypoints) == 0:
            continue
        assert len(keypoints) == len(keypoints_labels), (
            f"KeypointsLabels  lengthmismatch: {keypoints=}, {keypoints_labels=}"
        )
        for keypoint, keypointlabel in zip(keypoints, keypoints_labels):
            x, y = keypoint[0], keypoint[1]
            keypoints_labels_json.append({
                "original_width": im_width,
                "original_height": im_height,
                "image_rotation": 0,
                "value": {
                    "x": x * 100 / im_width,
                    "y": y * 100 / im_height,
                    "width": keypoints_width,
                    "keypointlabels": [keypointlabel]
                },
                "from_name": from_name,
                "to_name": to_name,
                "type": "keypointlabels"
            })
    return keypoints_labels_json


def parse_keypoint_label_to_keypoint(
    keypoint_label: Dict
) -> list:
    original_height = keypoint_label['original_height']
    original_width = keypoint_label['original_width']
    x, y = keypoint_label['value']['x'], keypoint_label['value']['y']
    x = x / 100 * original_width
    y = y / 100 * original_height
    x = round(x)
    y = round(y)
    label = keypoint_label['value']['keypointlabels'][0]
    return [x, y], label


def convert_image_data_to_annotation(
    image_data: ImageData,
    to_name: str,
    bboxes_from_name: Optional[str] = None,
    label_from_name: Optional[str] = None,
    keypoints_from_name: Optional[str] = None,
    keypoints_labels: Optional[List[str]] = None,
    keypoints_width: float = 0.8,
) -> List[Dict[str, Any]]:

    assert (keypoints_from_name is not None and keypoints_labels is not None) or (
        keypoints_from_name is None and keypoints_labels is None
    )

    im_width, im_height = image_data.get_image_size(force_update_meta=True)
    annotations = []
    if label_from_name is not None and image_data.label is not None:
        annotations.append({
            "value": {
                "choices": [image_data.label]
            },
            "from_name": label_from_name,
            "to_name": to_name,
            "type": "choices"
        })

    if bboxes_from_name is not None:
        for bbox_idx, bbox_data in enumerate(image_data.bboxes_data):
            annotations.append({
                "id": f"bbox{bbox_idx}",
                "original_width": im_width,
                "original_height": im_height,
                "image_rotation": 0,
                "value": {
                    "x": bbox_data.xmin / im_width * 100,
                    "y": bbox_data.ymin / im_height * 100,
                    "width": (bbox_data.xmax - bbox_data.xmin) / im_width * 100,
                    "height": (bbox_data.ymax - bbox_data.ymin) / im_height * 100,
                    "rotation": 0,
                    "rectanglelabels": [bbox_data.label]
                },
                "from_name": bboxes_from_name,
                "to_name": to_name,
                "type": "rectanglelabels",
            })
            if keypoints_from_name is not None:
                if len(bbox_data.keypoints) == 0:
                    continue
                assert len(bbox_data.keypoints) == len(keypoints_labels), (
                    f"KeypointsLabels length mismatch: {bbox_data.keypoints=}, {keypoints_labels=}"
                )

                for kp_idx, (keypoint, keypointlabel) in enumerate(zip(bbox_data.keypoints, keypoints_labels)):
                    x, y = keypoint[0], keypoint[1]
                    annotations.append({
                        "id": f"bbox{bbox_idx}_kp{kp_idx}",
                        "original_width": im_width,
                        "original_height": im_height,
                        "image_rotation": 0,
                        "value": {
                            "x": x * 100 / im_width,
                            "y": y * 100 / im_height,
                            "width": keypoints_width,
                            "keypointlabels": [keypointlabel]
                        },
                        "from_name": keypoints_from_name,
                        "to_name": to_name,
                        "type": "keypointlabels"
                    })
                    annotations.append({
                        'from_id': f"bbox{bbox_idx}_kp{kp_idx}",
                        'to_id': f"bbox{bbox_idx}",
                        'type': 'relation',
                        'direction': 'bi'
                    })

    if len(image_data.keypoints) > 0 and keypoints_from_name is not None:
        assert len(image_data.keypoints) == len(keypoints_labels), (
            f"KeypointsLabels mismatch: {image_data.keypoints=}, {keypoints_labels=}"
        )
        for kp_idx, (keypoint, keypointlabel) in enumerate(zip(image_data.keypoints, keypoints_labels)):
            x, y = keypoint[0], keypoint[1]
            annotations.append({
                "id": f"kp{kp_idx}",
                "original_width": im_width,
                "original_height": im_height,
                "image_rotation": 0,
                "value": {
                    "x": x * 100 / im_width,
                    "y": y * 100 / im_height,
                    "width": keypoints_width,
                    "keypointlabels": [keypointlabel]
                },
                "from_name": keypoints_from_name,
                "to_name": to_name,
                "type": "keypointlabels"
            })

    return annotations


def convert_annotation_to_image_data(
    annotation: List[Dict[str, Any]],
    bboxes_from_name: Optional[str] = None,
    label_from_name: Optional[str] = None,  # must be one, attended to image
    keypoints_from_name: Optional[str] = None,
    keypoints_labels: Optional[List[str]] = None,
    image_path: Optional[str] = None
) -> ImageData:
    assert (keypoints_from_name is not None and keypoints_labels is not None) or (
        keypoints_from_name is None and keypoints_labels is None
    )

    bboxes_data = []
    id_to_bbox_data_idx = {}
    keypoints = []
    items_keypoints_labels = []
    id_to_keypoint_idx = {}
    bbox_id_to_keypoints_idxs_relation = {}
    keypoints_idxs_that_have_relation = set()
    keypoint_label_to_position = {}
    image_data_keypoints = []
    image_data_label = None

    for result in annotation['result']:
        if label_from_name is not None and result['from_name'] == bboxes_from_name:
            if len(result['value']['choices']) > 0:
                if image_data_label is not None:
                    raise ValueError(
                        f"Found duplicated choices: {result=} (previously choice is {image_data_label=})"
                    )
                image_data_label = result['value']['choices'][0]
        if bboxes_from_name is not None:
            if result['type'] == 'rectanglelabels' and result['from_name'] == bboxes_from_name:
                bboxes_data.append(parse_rectangle_labels_to_bbox_data(result))
                id_to_bbox_data_idx[result['id']] = len(bboxes_data) - 1

        if keypoints_from_name is not None:
            if result['type'] == 'keypointlabels' and result['from_name'] == keypoints_from_name:
                (x, y), kp_label = parse_keypoint_label_to_keypoint(result)
                items_keypoints_labels.append(kp_label)
                keypoints.append((x, y))
                id_to_keypoint_idx[result['id']] = len(keypoints) - 1

            elif result['type'] == 'relation':
                if result['from_id'] in id_to_keypoint_idx:
                    keypoint_idx = result['from_id']
                    bbox_idx = result['to_id']
                elif result['to_id'] in id_to_keypoint_idx:
                    keypoint_idx = result['to_id']
                    bbox_idx = result['from_id']
                else:
                    continue
                if bbox_idx not in bbox_id_to_keypoints_idxs_relation:
                    bbox_id_to_keypoints_idxs_relation[bbox_idx] = []
                bbox_id_to_keypoints_idxs_relation[bbox_idx].append(keypoint_idx)
                keypoints_idxs_that_have_relation.add(keypoint_idx)

    if keypoints_from_name is not None:
        keypoint_label_to_position = {label: idx for idx, label in enumerate(keypoints_labels)}

    if bboxes_from_name is not None and keypoints_from_name is not None:
        for bbox_id, keypoints_idxs in bbox_id_to_keypoints_idxs_relation.items():
            if len(keypoints_idxs) > 0:
                assert len(keypoints_idxs) == len(keypoints_labels), (
                    f"KeypointsLabels length mismatch: {keypoints_idxs=}, {keypoints_labels=}"
                )
            item_keypoints = [keypoints[id_to_keypoint_idx[keypoint_idx]] for keypoint_idx in keypoints_idxs]
            item_keypoints_labels = [items_keypoints_labels[id_to_keypoint_idx[keypoint_idx]] for keypoint_idx in keypoints_idxs]
            item_keypoints_labels_positions = list(map(keypoint_label_to_position.get, item_keypoints_labels))
            item_sorted_idxs = np.argsort(item_keypoints_labels_positions)

            bboxes_data[id_to_bbox_data_idx[bbox_id]].keypoints = np.array(item_keypoints)[item_sorted_idxs]

    if keypoints_from_name is not None:
        keypoints_attended_to_image_data = []
        for keypoint_id in id_to_keypoint_idx:
            if keypoint_id not in keypoints_idxs_that_have_relation:
                keypoints_attended_to_image_data.append(keypoint_id)

        if len(keypoints_attended_to_image_data) > 0:
            assert len(keypoints_attended_to_image_data) == len(keypoints_labels), (
                f"KeypointsLabels length mismatch: {keypoints_attended_to_image_data=}, {keypoints_labels=}"
            )
        item_keypoints = [
            keypoints[id_to_keypoint_idx[keypoint_idx]]
            for keypoint_idx in keypoints_attended_to_image_data
        ]
        item_keypoints_labels = [
            items_keypoints_labels[id_to_keypoint_idx[keypoint_idx]]
            for keypoint_idx in keypoints_attended_to_image_data
        ]
        item_keypoints_labels_positions = list(map(keypoint_label_to_position.get, item_keypoints_labels))
        item_sorted_idxs = np.argsort(item_keypoints_labels_positions)
        image_data_keypoints = np.array(item_keypoints)[image_data_keypoints]

    return ImageData(
        image_path=image_path,
        bboxes_data=bboxes_data,
        label=image_data_label,
        keypoints=image_data_keypoints
    )
