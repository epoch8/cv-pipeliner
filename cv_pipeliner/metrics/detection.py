import contextlib
import os
from typing import Dict, List, Type

import numpy as np
import pandas as pd

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.metrics.image_data_matching import ImageDataMatching


def count_coco_metrics(
    true_images_data: List[ImageData], raw_pred_images_data: List[ImageData] = None, class_names: List[str] = None
) -> Dict:
    from object_detection.core.standard_fields import (
        DetectionResultFields,
        InputDataFields,
    )
    from object_detection.metrics import coco_evaluation

    if class_names is None:
        cocoevaluator = coco_evaluation.CocoDetectionEvaluator(categories=[{"id": 1, "name": "Label"}])
    else:
        cocoevaluator = coco_evaluation.CocoDetectionEvaluator(
            categories=[{"id": i + 1, "name": class_name} for i, class_name in enumerate(class_names)]
        )

    for i, (true_image_data, raw_pred_image_data) in enumerate(zip(true_images_data, raw_pred_images_data)):
        true_bboxes_data = true_image_data.bboxes_data
        raw_pred_bboxes_data = raw_pred_image_data.bboxes_data

        true_bboxes = [
            (true_bbox_data.ymin, true_bbox_data.xmin, true_bbox_data.ymax, true_bbox_data.xmax)
            for true_bbox_data in true_bboxes_data
        ]
        raw_pred_bboxes = [
            (raw_pred_bbox_data.ymin, raw_pred_bbox_data.xmin, raw_pred_bbox_data.ymax, raw_pred_bbox_data.xmax)
            for raw_pred_bbox_data in raw_pred_bboxes_data
        ]
        raw_pred_scores = [raw_pred_bbox_data.detection_score for raw_pred_bbox_data in raw_pred_bboxes_data]

        groundtruth_boxes = np.array(true_bboxes, dtype=np.float32).reshape(-1, 4)
        if class_names is None:
            groundtruth_classes = np.array([1] * len(true_bboxes))
        else:
            groundtruth_classes = np.array([class_names.index(bbox_data.label) + 1 for bbox_data in true_bboxes_data])
        groundtruth_dict = {
            InputDataFields.groundtruth_boxes: groundtruth_boxes,
            InputDataFields.groundtruth_classes: groundtruth_classes,
        }
        cocoevaluator.add_single_ground_truth_image_info(image_id=i, groundtruth_dict=groundtruth_dict)

        detection_boxes = np.array(raw_pred_bboxes, dtype=np.float32).reshape(-1, 4)
        detection_scores = np.array(raw_pred_scores, dtype=np.float32)
        if class_names is None:
            detection_classes = np.array([1] * len(detection_boxes))
        else:
            detection_classes = np.array([class_names.index(bbox_data.label) + 1 for bbox_data in raw_pred_bboxes_data])
        detections_dict = {
            DetectionResultFields.detection_boxes: detection_boxes,
            DetectionResultFields.detection_scores: detection_scores,
            DetectionResultFields.detection_classes: detection_classes,
        }
        cocoevaluator.add_single_detected_image_info(image_id=i, detections_dict=detections_dict)

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        coco_metrics = cocoevaluator.evaluate()
    return coco_metrics


df_detection_metrics_columns = ["value"]


def get_df_detection_metrics(
    true_images_data: List[ImageData],
    pred_images_data: List[ImageData],
    minimum_iou: float,
    raw_pred_images_data: List[ImageData] = None,
    image_data_matching_class: Type[ImageDataMatching] = ImageDataMatching,
) -> pd.DataFrame:
    """
    Returns detection metrics (precision, recall, f1_score, mAP).
    """

    assert len(true_images_data) == len(pred_images_data)

    images_data_matchings = [
        image_data_matching_class(true_image_data, pred_image_data, minimum_iou)
        for true_image_data, pred_image_data in zip(true_images_data, pred_images_data)
    ]
    TP = np.sum([image_data_matching.get_detection_TP() for image_data_matching in images_data_matchings])
    FP = np.sum([image_data_matching.get_detection_FP() for image_data_matching in images_data_matchings])
    FN = np.sum([image_data_matching.get_detection_FN() for image_data_matching in images_data_matchings])
    iou_mean = np.mean(
        [
            bbox_data_matching.iou
            for image_data_matching in images_data_matchings
            for bbox_data_matching in image_data_matching.bboxes_data_matchings
            if bbox_data_matching.iou is not None
        ]
    )
    accuracy = TP / max(TP + FN + FN, 1e-6)
    precision = TP / max(TP + FP, 1e-6)
    recall = TP / max(TP + FN, 1e-6)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
    coco_metrics = {}
    if raw_pred_images_data is not None:
        try:
            coco_metrics = count_coco_metrics(true_images_data, raw_pred_images_data)
        except ModuleNotFoundError:
            pass

    df_detection_metrics = pd.DataFrame(
        {
            "images_support": len(true_images_data),
            "support": [TP + FN],
            "TP": [TP],
            "FP": [FP],
            "FN": [FN],
            "iou_mean": [iou_mean],
            "accuracy": [accuracy],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1_score],
            **coco_metrics,
        },
        dtype=object,
    ).T
    df_detection_metrics.columns = df_detection_metrics_columns

    return df_detection_metrics
