import os
import contextlib
from typing import List, Dict

import pandas as pd
import numpy as np

from object_detection.metrics import coco_evaluation
from object_detection.core.standard_fields import InputDataFields, \
                                                  DetectionResultFields

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.metrics.image_data_matching import ImageDataMatching


def count_coco_metrics(
    true_images_data: List[ImageData],
    raw_pred_images_data: List[ImageData] = None
) -> Dict:
    cocoevaluator = coco_evaluation.CocoDetectionEvaluator(
        categories=[{
            'id': 1,
            'name': 'Label'
        }]
    )

    for i, (true_image_data, raw_pred_image_data) in enumerate(zip(
        true_images_data, raw_pred_images_data
    )):
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
        raw_pred_scores = [
            raw_pred_bbox_data.detection_score
            for raw_pred_bbox_data in raw_pred_bboxes_data
        ]

        groundtruth_boxes = np.array(true_bboxes, dtype=np.float32).reshape(-1, 4)
        groundtruth_classes = np.array([1] * len(true_bboxes))
        groundtruth_dict = {
            InputDataFields.groundtruth_boxes: groundtruth_boxes,
            InputDataFields.groundtruth_classes: groundtruth_classes
        }
        cocoevaluator.add_single_ground_truth_image_info(
            image_id=i,
            groundtruth_dict=groundtruth_dict
        )

        detection_boxes = np.array(raw_pred_bboxes, dtype=np.float32).reshape(-1, 4)
        detection_scores = np.array(raw_pred_scores, dtype=np.float32)
        detection_classes = np.array([1] * len(detection_boxes))
        detections_dict = {
            DetectionResultFields.detection_boxes: detection_boxes,
            DetectionResultFields.detection_scores: detection_scores,
            DetectionResultFields.detection_classes: detection_classes
        }
        cocoevaluator.add_single_detected_image_info(
            image_id=i,
            detections_dict=detections_dict
        )

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        coco_metrics = cocoevaluator.evaluate()
    return coco_metrics


def get_df_detection_metrics(
    true_images_data: List[ImageData],
    pred_images_data: List[ImageData],
    minimum_iou: float,
    raw_pred_images_data: List[ImageData] = None,
) -> pd.DataFrame:
    '''
    Returns detection metrics (precision, recall, f1_score, mAP).
    '''

    assert len(true_images_data) == len(pred_images_data)

    images_data_matchings = [
        ImageDataMatching(true_image_data, pred_image_data, minimum_iou)
        for true_image_data, pred_image_data in zip(true_images_data, pred_images_data)
    ]
    TP = np.sum([image_data_matching.get_detection_TP() for image_data_matching in images_data_matchings])
    FP = np.sum([image_data_matching.get_detection_FP() for image_data_matching in images_data_matchings])
    FN = np.sum([image_data_matching.get_detection_FN() for image_data_matching in images_data_matchings])
    iou_mean = np.mean([
        bbox_data_matching.iou
        for image_data_matching in images_data_matchings
        for bbox_data_matching in image_data_matching.bboxes_data_matchings
        if bbox_data_matching.iou is not None
    ])
    accuracy = TP / max(TP + FN + FN, 1e-6)
    precision = TP / max(TP + FP, 1e-6)
    recall = TP / max(TP + FN, 1e-6)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
    if raw_pred_images_data is not None:
        coco_metrics = count_coco_metrics(true_images_data, raw_pred_images_data)
    else:
        coco_metrics = {}

    df_detection_metrics = pd.DataFrame({
        'TP': [TP],
        'FP': [FP],
        'FN': [FN],
        'iou_mean': [iou_mean],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1_score],
        **coco_metrics
    }, dtype=object).T
    df_detection_metrics.columns = ['value']

    return df_detection_metrics


def get_df_detection_recall_per_class(
    true_images_data: List[ImageData],
    pred_images_data: List[ImageData],
    minimum_iou: float
) -> pd.DataFrame:
    '''
    Returns detection recall per every class, when labels are given in true_images_data.
    '''
    assert len(true_images_data) == len(pred_images_data)

    images_data_matchings = [
        ImageDataMatching(true_image_data, pred_image_data, minimum_iou)
        for true_image_data, pred_image_data in zip(true_images_data, pred_images_data)
    ]
    true_labels = np.array([
        bbox_data.label
        for image_data in true_images_data
        for bbox_data in image_data.bboxes_data
    ])
    class_names = np.unique(true_labels)

    detection_metrics_recall_per_class = {}
    for class_name in class_names:
        TP_by_class_name = np.sum(
            image_data_matching.get_detection_TP(filter_by_label=class_name)
            for image_data_matching in images_data_matchings
        )
        FN_by_class_name = np.sum(
            image_data_matching.get_detection_FN(filter_by_label=class_name)
            for image_data_matching in images_data_matchings
        )
        recall = TP_by_class_name / max(TP_by_class_name + FN_by_class_name, 1e-6)
        detection_metrics_recall_per_class[class_name] = {
            'support': TP_by_class_name + FN_by_class_name,
            'TP': TP_by_class_name,
            'FN': FN_by_class_name,
            'recall': recall,
        }
    supports = np.array([
        detection_metrics_recall_per_class[class_name]['support']
        for class_name in class_names
    ])
    recalls = np.array([
        detection_metrics_recall_per_class[class_name]['recall']
        for class_name in class_names
    ])
    macro_average_recall = np.mean(recalls)
    weighted_average_recall = np.average(recalls, weights=supports)
    sum_support = np.sum(supports)
    sum_TP = np.sum([
        detection_metrics_recall_per_class[class_name]['TP']
        for class_name in class_names
    ])
    sum_FN = np.sum([
        detection_metrics_recall_per_class[class_name]['FN']
        for class_name in class_names
    ])
    detection_metrics_recall_per_class['macro average'] = {
        "support": sum_support,
        'TP': sum_TP,
        'FN': sum_FN,
        "recall": macro_average_recall,
    }
    detection_metrics_recall_per_class['weighted average'] = {
        "support": sum_support,
        'TP': sum_TP,
        'FN': sum_FN,
        "recall": weighted_average_recall,
    }

    df_detection_recall_per_class = pd.DataFrame(detection_metrics_recall_per_class, dtype=object).T
    df_detection_recall_per_class = df_detection_recall_per_class[['support', 'TP', 'FN', 'recall']]
    df_detection_recall_per_class.sort_values(by='support', ascending=False, inplace=True)

    return df_detection_recall_per_class
