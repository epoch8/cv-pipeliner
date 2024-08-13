from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.utils.images import rotate_point
from cv_pipeliner.utils.images_datas import combine_mask_polygons_to_one_polygon


def parse_rectangle_labels_to_bbox_data(rectangle_label: Dict) -> BboxData:
    original_height = rectangle_label["original_height"]
    original_width = rectangle_label["original_width"]
    height = rectangle_label["value"]["height"]
    width = rectangle_label["value"]["width"]
    xmin = rectangle_label["value"]["x"]
    ymin = rectangle_label["value"]["y"]
    label = rectangle_label["value"]["rectanglelabels"][0]
    xmax = xmin + width
    ymax = ymin + height
    xmin = max(0, min(original_width - 1, xmin / 100 * original_width))
    ymin = max(0, min(original_height - 1, ymin / 100 * original_height))
    xmax = max(0, min(original_width - 1, xmax / 100 * original_width))
    ymax = max(0, min(original_height - 1, ymax / 100 * original_height))
    angle = rectangle_label["value"]["rotation"]
    points = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    rotated_points = [rotate_point(x=x, y=y, cx=xmin, cy=ymin, angle=angle) for (x, y) in points]
    xmin = max(0, min([x for (x, y) in rotated_points]))
    ymin = max(0, min([y for (x, y) in rotated_points]))
    xmax = max([x for (x, y) in rotated_points])
    ymax = max([y for (x, y) in rotated_points])
    xmin = max(0, min(original_width - 1, xmin))
    ymin = max(0, min(original_height - 1, ymin))
    xmax = max(0, min(original_width - 1, xmax))
    ymax = max(0, min(original_height - 1, ymax))
    bbox_data = BboxData(
        xmin=round(xmin),
        ymin=round(ymin),
        xmax=round(xmax),
        ymax=round(ymax),
        label=label,
        meta_height=original_height,
        meta_width=original_width,
    )
    return bbox_data


def convert_image_data_to_rectangle_labels(
    image_data: ImageData,
    from_name: str,
    to_name: str,
) -> Dict:
    im_width, im_height = image_data.get_image_size(exif_transpose=False)
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
                "rectanglelabels": [bbox_data.label],
            },
            "from_name": from_name,
            "to_name": to_name,
            "type": "rectanglelabels",
        }
        rectangle_labels.append(rectangle_label)
    return rectangle_labels


def parse_polygon_label_to_bbox_data(
    polygon_label: Dict,
    keypoints_from_name: Optional[str],
    mask_from_name: Optional[str],
) -> BboxData:
    original_height = polygon_label["original_height"]
    original_width = polygon_label["original_width"]
    keypoints = []
    for x, y in polygon_label["value"]["points"]:
        x = x / 100 * polygon_label["original_width"]
        y = y / 100 * polygon_label["original_height"]
        keypoints.append([round(max(0, min(original_width - 1, x))), round(max(0, min(original_height - 1, y)))])
    keypoints = np.array(keypoints)
    bbox_data = BboxData(
        xmin=round(np.min(keypoints[:, 0])),
        ymin=round(np.min(keypoints[:, 1])),
        xmax=round(np.max(keypoints[:, 0])),
        ymax=round(np.max(keypoints[:, 1])),
        keypoints=(np.round(keypoints) if polygon_label["from_name"] == keypoints_from_name else None),
        mask=([keypoints] if polygon_label["from_name"] == mask_from_name else None),
        label=polygon_label["value"]["polygonlabels"][0],
    )
    return bbox_data


def parse_polygon_label_to_mask(polygon_label: Dict):
    original_height = polygon_label["original_height"]
    original_width = polygon_label["original_width"]
    mask = []
    for x, y in polygon_label["value"]["points"]:
        x = x / 100 * polygon_label["original_width"]
        y = y / 100 * polygon_label["original_height"]
        mask.append([round(max(0, min(original_width - 1, x))), round(max(0, min(original_height - 1, y)))])
    labels = polygon_label["value"]["polygonlabels"]
    label = labels[0] if len(labels) == 1 else None
    return [mask], label


def convert_image_data_to_polygon_label(
    image_data: ImageData,
    from_name: str,
    to_name: str,
    polygonlabels: str,
) -> Dict:
    im_width, im_height = image_data.get_image_size(exif_transpose=False)
    rectangle_labels = []
    for bbox_data in image_data.bboxes_data:
        rectangle_labels.append(
            {
                "original_width": im_width,
                "original_height": im_height,
                "image_rotation": 0,
                "value": {
                    "points": [[x * 100 / im_width, y * 100 / im_height] for x, y in bbox_data.keypoints],
                    "polygonlabels": [polygonlabels],
                },
                "from_name": from_name,
                "to_name": to_name,
                "type": "polygonlabels",
            }
        )
    return rectangle_labels


def convert_image_data_to_keypoint_label(
    image_data: ImageData, from_name: str, to_name: str, keypoints_labels: List[str], keypoints_width: float = 0.8
) -> Dict:
    im_width, im_height = image_data.get_image_size(exif_transpose=False)
    keypoints_labels_json = []
    keypoints_to_be_added = [image_data.keypoints] + [bbox_data.keypoints for bbox_data in image_data.bboxes_data]
    for keypoints in keypoints_to_be_added:
        if len(keypoints) == 0:
            continue
        assert len(keypoints) == len(
            keypoints_labels
        ), f"KeypointsLabels  lengthmismatch: {keypoints=}, {keypoints_labels=}"
        for keypoint, keypointlabel in zip(keypoints, keypoints_labels):
            x, y = keypoint[0], keypoint[1]
            keypoints_labels_json.append(
                {
                    "original_width": im_width,
                    "original_height": im_height,
                    "image_rotation": 0,
                    "value": {
                        "x": x * 100 / im_width,
                        "y": y * 100 / im_height,
                        "width": keypoints_width,
                        "keypointlabels": [keypointlabel],
                    },
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "keypointlabels",
                }
            )
    return keypoints_labels_json


def parse_keypoint_label_to_keypoint(keypoint_label: Dict) -> list:
    original_height = keypoint_label["original_height"]
    original_width = keypoint_label["original_width"]
    x, y = keypoint_label["value"]["x"], keypoint_label["value"]["y"]
    x = x / 100 * original_width
    y = y / 100 * original_height
    x = round(x)
    y = round(y)
    labels = keypoint_label["value"]["keypointlabels"]
    label = labels[0] if len(labels) == 1 else None
    return [x, y], label


def convert_image_data_to_annotation(
    image_data: ImageData,
    to_name: str,
    bboxes_from_name: Optional[str] = None,
    label_from_name: Optional[str] = None,
    keypoints_from_name: Optional[str] = None,
    keypoints_labels: Optional[List[str]] = None,
    mask_from_name: Optional[str] = None,
    keypoints_width: float = 0.8,
) -> List[Dict[str, Any]]:
    # При импорте разметки в LS не учитываются Exif теги, но при экспорте оно учитывается
    im_width, im_height = image_data.get_image_size(exif_transpose=False)
    annotations = []
    if label_from_name is not None and image_data.label is not None:
        annotations.append(
            {
                "value": {"choices": [image_data.label]},
                "from_name": label_from_name,
                "to_name": to_name,
                "type": "choices",
            }
        )

    if bboxes_from_name is not None:
        for bbox_idx, bbox_data in enumerate(image_data.bboxes_data):
            annotations.append(
                {
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
                        "rectanglelabels": [bbox_data.label],
                    },
                    "from_name": bboxes_from_name,
                    "to_name": to_name,
                    "type": "rectanglelabels",
                }
            )
            if keypoints_from_name is not None:
                if len(bbox_data.keypoints) == 0:
                    continue
                for kp_idx, keypoint in enumerate(bbox_data.keypoints):
                    if keypoints_labels is not None:
                        assert len(keypoints_labels) == len(bbox_data.keypoints)
                    x, y = keypoint[0], keypoint[1]
                    annotations.append(
                        {
                            "id": f"bbox{bbox_idx}_kp{kp_idx}",
                            "original_width": im_width,
                            "original_height": im_height,
                            "image_rotation": 0,
                            "value": {
                                "x": x * 100 / im_width,
                                "y": y * 100 / im_height,
                                "width": keypoints_width,
                                "keypointlabels": ([keypoints_labels[kp_idx]] if keypoints_labels is not None else []),
                            },
                            "from_name": keypoints_from_name,
                            "to_name": to_name,
                            "type": "keypointlabels",
                        }
                    )
                    annotations.append(
                        {
                            "from_id": f"bbox{bbox_idx}_kp{kp_idx}",
                            "to_id": f"bbox{bbox_idx}",
                            "type": "relation",
                            "direction": "bi",
                        }
                    )
            if mask_from_name is not None:
                assert len(bbox_data.mask) == 1
                annotations.append(
                    {
                        "id": f"bbox{bbox_idx}_mask",
                        "original_width": im_width,
                        "original_height": im_height,
                        "image_rotation": 0,
                        "value": {
                            "points": [
                                [x * 100 / im_width, y * 100 / im_height]
                                for x, y in combine_mask_polygons_to_one_polygon(bbox_data.mask)
                            ],
                            "polygonlabels": [bbox_data.label],
                        },
                        "from_name": mask_from_name,
                        "to_name": to_name,
                        "type": "polygonlabels",
                    }
                )
                annotations.append(
                    {
                        "from_id": f"bbox{bbox_idx}_mask",
                        "to_id": f"bbox{bbox_idx}",
                        "type": "relation",
                        "direction": "bi",
                    }
                )

    if len(image_data.keypoints) > 0 and keypoints_from_name is not None:
        if keypoints_labels is not None:
            assert len(image_data.keypoints) == len(
                keypoints_labels
            ), f"KeypointsLabels mismatch: {image_data.keypoints=}, {keypoints_labels=}"
        for kp_idx, (keypoint, keypointlabel) in enumerate(zip(image_data.keypoints, keypoints_labels)):
            x, y = keypoint[0], keypoint[1]
            annotations.append(
                {
                    "id": f"kp{kp_idx}",
                    "original_width": im_width,
                    "original_height": im_height,
                    "image_rotation": 0,
                    "value": {
                        "x": x * 100 / im_width,
                        "y": y * 100 / im_height,
                        "width": keypoints_width,
                        "keypointlabels": [keypointlabel],
                    },
                    "from_name": keypoints_from_name,
                    "to_name": to_name,
                    "type": "keypointlabels",
                }
            )

    if len(image_data.mask) > 0 and mask_from_name is not None:
        annotations.append(
            {
                "id": "mask",
                "original_width": im_width,
                "original_height": im_height,
                "image_rotation": 0,
                "value": {
                    "points": [
                        [x * 100 / im_width, y * 100 / im_height]
                        for x, y in combine_mask_polygons_to_one_polygon(image_data.mask)
                    ],
                    "polygonlabels": [image_data.label],
                },
                "from_name": mask_from_name,
                "to_name": to_name,
                "type": "polygonlabels",
            }
        )

    return annotations


def parse_result(
    result: Dict[str, Any],
    bboxes_from_name: Optional[str],
    label_from_name: Optional[str],
    keypoints_from_name: Optional[str],
    mask_from_name: Optional[str],
) -> Optional[Union[str, Tuple[str, Dict[str, Any], str], Tuple[str, Tuple[float, float], str, str]]]:
    from_name = result.get("from_name", None)
    if from_name is None:
        return None
    if label_from_name and result["from_name"] == label_from_name:
        if len(result["value"]["choices"]) > 0:
            return result["value"]["choices"][0]
    if bboxes_from_name and result["from_name"] == bboxes_from_name:
        if result["type"] == "rectanglelabels":
            return "bbox", parse_rectangle_labels_to_bbox_data(result), result["id"]
        elif result["type"] == "polygonlabels":
            return "bbox", parse_polygon_label_to_bbox_data(result, keypoints_from_name, mask_from_name), result["id"]
    if keypoints_from_name:
        if result["type"] == "keypointlabels" and result["from_name"] == keypoints_from_name:
            (x, y), kp_label = parse_keypoint_label_to_keypoint(result)
            return "keypoint", (x, y), kp_label, result["id"]
    if mask_from_name:
        if result["type"] == "polygonlabels" and result["from_name"] == mask_from_name:
            mask, mask_label = parse_polygon_label_to_mask(result)
            if result["from_name"] == mask_from_name:
                return "mask", mask, mask_label, result["id"]
    return None


def handle_relations(
    result: Dict[str, Any],
    id_to_keypoint_idx: Dict[str, int],
    id_to_mask_idx: Dict[str, int],
    bbox_id_to_keypoints_idxs_relation: Dict[str, List[str]],
    keypoints_idxs_that_have_relation: set,
    bbox_id_to_masks_idxs_relation: Dict[str, List[str]],
    masks_idxs_that_have_relation: set,
) -> None:
    keypoint_idx: Optional[str] = None
    mask_idx: Optional[str] = None
    if result["from_id"] in id_to_keypoint_idx:
        keypoint_idx = result["from_id"]
        bbox_idx = result["to_id"]
    elif result["to_id"] in id_to_keypoint_idx:
        keypoint_idx = result["to_id"]
        bbox_idx = result["from_id"]
    if result["from_id"] in id_to_mask_idx:
        mask_idx = result["from_id"]
        bbox_idx = result["to_id"]
    elif result["to_id"] in id_to_mask_idx:
        mask_idx = result["to_id"]
        bbox_idx = result["from_id"]
    if keypoint_idx is not None:
        bbox_id_to_keypoints_idxs_relation.setdefault(bbox_idx, []).append(keypoint_idx)
        keypoints_idxs_that_have_relation.add(keypoint_idx)
    if mask_idx is not None:
        bbox_id_to_masks_idxs_relation.setdefault(bbox_idx, []).append(mask_idx)
        masks_idxs_that_have_relation.add(mask_idx)


def process_annotations(
    annotation: List[Dict[str, Any]],
    bboxes_from_name: Optional[str],
    label_from_name: Optional[str],
    keypoints_from_name: Optional[str],
    mask_from_name: Optional[str],
) -> Tuple[
    List[Dict[str, Any]],
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[str],
    List[str],
    Optional[str],
]:
    bboxes_data: List[Dict[str, Any]] = []
    id_to_bbox_data_idx: Dict[str, int] = {}
    keypoints: List[Tuple[float, float]] = []
    masks: List[Tuple[float, float]] = []
    items_keypoints_labels: List[str] = []
    items_mask_labels: List[str] = []
    id_to_keypoint_idx: Dict[str, int] = {}
    id_to_mask_idx: Dict[str, int] = {}
    bbox_id_to_keypoints_idxs_relation: Dict[str, List[str]] = {}
    bbox_id_to_masks_idxs_relation: Dict[str, List[str]] = {}
    keypoints_idxs_that_have_relation: set = set()
    masks_idxs_that_have_relation: set = set()
    image_data_label: Optional[str] = None

    for result in annotation["result"]:
        parsed_result = parse_result(result, bboxes_from_name, label_from_name, keypoints_from_name, mask_from_name)
        if parsed_result:
            if isinstance(parsed_result, str):
                if image_data_label:
                    raise ValueError(f"Found duplicated choices: {result=} (previously choice is {image_data_label=})")
                image_data_label = parsed_result
            elif parsed_result[0] == "bbox":
                bboxes_data.append(parsed_result[1])
                id_to_bbox_data_idx[parsed_result[2]] = len(bboxes_data) - 1
            elif parsed_result[0] == "keypoint":
                keypoints.append(parsed_result[1])
                items_keypoints_labels.append(parsed_result[2])
                id_to_keypoint_idx[parsed_result[3]] = len(keypoints) - 1
            elif parsed_result[0] == "mask":
                masks.append(parsed_result[1])
                items_mask_labels.append(parsed_result[2])
                id_to_mask_idx[parsed_result[3]] = len(masks) - 1

        if result["type"] == "relation":
            handle_relations(
                result,
                id_to_keypoint_idx,
                id_to_mask_idx,
                bbox_id_to_keypoints_idxs_relation,
                keypoints_idxs_that_have_relation,
                bbox_id_to_masks_idxs_relation,
                masks_idxs_that_have_relation,
            )

    return (
        bboxes_data,
        keypoints,
        masks,
        items_keypoints_labels,
        items_mask_labels,
        image_data_label,
    )


def sort_items_by_labels(
    items: List[Union[Tuple[float, float], Tuple[float, float]]], labels: List[str], label_to_position: Dict[str, int]
) -> np.ndarray:
    item_labels_positions = list(map(label_to_position.get, labels))
    item_sorted_idxs = np.argsort(item_labels_positions)
    return np.array(items)[item_sorted_idxs]


def convert_annotation_to_image_data(
    annotation: List[Dict[str, Any]],
    bboxes_from_name: Optional[str] = None,
    label_from_name: Optional[str] = None,
    keypoints_from_name: Optional[str] = None,
    keypoints_labels: Optional[List[str]] = None,
    mask_from_name: Optional[str] = None,
    mask_labels: Optional[List[str]] = None,
    image_path: Optional[str] = None,
) -> ImageData:
    (
        bboxes_data,
        keypoints,
        masks,
        items_keypoints_labels,
        items_mask_labels,
        image_data_label,
    ) = process_annotations(annotation, bboxes_from_name, label_from_name, keypoints_from_name, mask_from_name)

    if keypoints_labels is not None:
        label_to_position = {label: idx for idx, label in enumerate(keypoints_labels)}
        keypoints = sort_items_by_labels(keypoints, items_keypoints_labels, label_to_position)

    if mask_labels is not None:
        label_to_position = {label: idx for idx, label in enumerate(mask_labels)}
        masks = sort_items_by_labels(masks, items_mask_labels, label_to_position)

    return ImageData(
        image_path=image_path, bboxes_data=bboxes_data, label=image_data_label, keypoints=keypoints, mask=masks
    )
