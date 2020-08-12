import os
import contextlib

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


from object_detection.metrics import coco_evaluation
from object_detection.core.standard_fields import InputDataFields, \
                                                  DetectionResultFields


def iou_bbox(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    ymins = max(bbox1[0], bbox2[0])
    xmins = max(bbox1[1], bbox2[1])
    ymaxs = min(bbox1[2], bbox2[2])
    xmaxs = min(bbox1[3], bbox2[3])
    interArea = max(0, ymaxs - ymins + 1) * max(0, xmaxs - xmins + 1)
    bbox1Area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2Area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    val_iou = interArea / float(bbox1Area + bbox2Area - interArea + 1e-9)

    return val_iou


def find_best_bbox_by_iou(true_bbox: Tuple[int, int, int, int],
                          pred_bboxes: List[Tuple[int, int, int, int]],
                          minimum_iou: float) -> int:
    if len(pred_bboxes) == 0:
        return None
    bboxes_iou = [iou_bbox(true_bbox, pred_bbox)
                  for pred_bbox in pred_bboxes]
    best_pred_bbox_idx = np.argmax(bboxes_iou)

    best_iou = bboxes_iou[best_pred_bbox_idx]
    if best_iou > minimum_iou:
        return best_pred_bbox_idx
    else:
        return None


def get_df_matching_for_one_item(
    true_bboxes: List[Tuple[int, int, int, int]],
    pred_bboxes: List[Tuple[int, int, int, int]],
    minimum_iou: float
) -> pd.DataFrame:
    true_bboxes_num = len(true_bboxes)
    pred_bboxes_num = len(pred_bboxes)
    remained_pred_bboxes = pred_bboxes.copy()
    items = []

    for true_bbox in true_bboxes:
        best_pred_bbox_idx = find_best_bbox_by_iou(
            true_bbox, remained_pred_bboxes, minimum_iou
        )
        if best_pred_bbox_idx is not None:
            best_pred_bbox = remained_pred_bboxes[best_pred_bbox_idx]
            item = {
                'true_bbox': tuple(true_bbox),
                'found': True,
                'pred_bbox': tuple(best_pred_bbox),
                'iou': iou_bbox(true_bbox, best_pred_bbox),
            }
            remained_pred_bboxes = np.delete(
                remained_pred_bboxes, best_pred_bbox_idx, axis=0
            )
        else:
            item = {
                'true_bbox': tuple(true_bbox),
                'found': False,
                'pred_bbox': None,
                'iou': 0.,
            }
        items.append(item)

    df_items = pd.DataFrame(items)

    TP = np.sum(df_items['found']) if len(df_items) > 0 else 0
    FP = len(remained_pred_bboxes)
    FN = len(true_bboxes) - TP

    df_matchings = pd.DataFrame({
        'items': [items],
        'TP': [TP],
        'FP': [FP],
        'FN': [FN],
        'true_bboxes_num': [true_bboxes_num],
        'pred_bboxes_num': [pred_bboxes_num]
    })
    return df_matchings


def get_df_detector_matchings(
    n_true_bboxes: List[List[Tuple[int, int, int, int]]],
    n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
    minimum_iou: float
) -> pd.DataFrame:

    assert len(n_true_bboxes) == len(n_pred_bboxes)

    detector_matchings = []
    for (true_bboxes,
            pred_bboxes) in zip(n_true_bboxes,
                                n_pred_bboxes):
        df_detector_matchings_one_item = get_df_matching_for_one_item(
            true_bboxes, pred_bboxes, minimum_iou
        )
        detector_matchings.append(df_detector_matchings_one_item)

    df_detector_matchings = pd.concat(detector_matchings,
                                      ignore_index=True)

    return df_detector_matchings


def get_df_all_bboxes_matchings(
    n_true_bboxes: List[List[Tuple[int, int, int, int]]],
    n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
    minimum_iou: float
) -> pd.DataFrame:

    assert len(n_true_bboxes) == len(n_pred_bboxes)

    n_true_bboxes = [
        [tuple(bbox) for bbox in sublist]
        for sublist in n_true_bboxes
    ]
    n_pred_bboxes = [
        [tuple(bbox) for bbox in sublist]
        for sublist in n_pred_bboxes
    ]

    df = get_df_detector_matchings(n_true_bboxes,
                                   n_pred_bboxes,
                                   minimum_iou)
    df['indexes'] = df.index
    df['true_coordinates'] = n_true_bboxes
    df['pred_coordinates'] = n_pred_bboxes

    df_true_coordinates = []
    for index, true_coordinates in zip(df['indexes'],
                                       df['true_coordinates']):
        for true_coordinate in true_coordinates:
            df_true_coordinates.append(
                {
                    'indexes': index,
                    'true_bbox': true_coordinate
                }
            )
    df_true_coordinates = pd.DataFrame(df_true_coordinates)

    def add_indexes_to_item(row):
        for item in row['items']:
            item['indexes'] = row['indexes']
        return row

    df['items'] = df.apply(add_indexes_to_item, axis=1)
    df_items = pd.DataFrame(
        [item for sublist in df['items'].tolist() for item in sublist]
    )

    df_all_predictions = []
    for indexes, pred_coordinate in zip(df['indexes'],
                                        df['pred_coordinates']):
        for pred_bbox in pred_coordinate:
            df_all_predictions.append(
                {
                    'indexes': indexes,
                    'pred_bbox': tuple(pred_bbox)
                }
            )

    df_all_predictions = pd.DataFrame(df_all_predictions).drop_duplicates()

    df_bboxes = pd.merge(df_true_coordinates, df_items)
    if len(df_all_predictions) == 0:
        df_all_predictions = pd.DataFrame(
            {
                'indexes': [0],
                'pred_bbox': [(0, 0, 0, 0)]
            }
        )
    df_bboxes = pd.merge(df_bboxes, df_all_predictions, how='left')
    df_bboxes['is_true_bbox'] = True

    assert len(df_bboxes) == len(df_true_coordinates)

    df_false_positive_bboxes = pd.merge(
        df_bboxes[['pred_bbox']],
        df_all_predictions,
        how='outer',
        indicator=True
    ).query('_merge == "right_only"')
    df_false_positive_bboxes = pd.merge(df_all_predictions,
                                        df_false_positive_bboxes)
    df_false_positive_bboxes['found'] = True
    df_false_positive_bboxes['is_true_bbox'] = False

    df_bboxes_all = df_bboxes.copy()
    df_bboxes_all['_merge'] = 'both'
    df_bboxes_all = pd.concat(
        [df_bboxes_all, df_false_positive_bboxes],
        axis=0, sort=False
    )
    df_bboxes_all = df_bboxes_all.reset_index(drop=True)
    df_bboxes_all.drop(columns=['_merge'], inplace=True)

    return df_bboxes_all


def count_coco_metrics(
    n_true_bboxes: List[List[Tuple[int, int, int, int]]],
    raw_n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
    raw_n_pred_scores: List[List[float]]
) -> Dict:
    cocoevaluator = coco_evaluation.CocoDetectionEvaluator(
        categories=[{
            'id': 1,
            'name': 'Label'
        }]
    )

    for i, (true_bboxes, pred_bboxes, pred_scores) in enumerate(zip(
        n_true_bboxes, raw_n_pred_bboxes, raw_n_pred_scores
    )):
        if len(true_bboxes) == 0:
            groundtruth_boxes = np.array([[]], dtype=np.float32).reshape(-1, 4)
            groundtruth_classes = np.array([])
        else:
            groundtruth_boxes = np.array(true_bboxes, dtype=np.float32)
            groundtruth_classes = np.array([1] * len(true_bboxes))
        groundtruth_dict = {
            InputDataFields.groundtruth_boxes: groundtruth_boxes,
            InputDataFields.groundtruth_classes: groundtruth_classes
        }

        cocoevaluator.add_single_ground_truth_image_info(
            image_id=i,
            groundtruth_dict=groundtruth_dict
        )

        detection_boxes = np.array(pred_bboxes, dtype=np.float32)
        detection_scores = np.array(pred_scores, dtype=np.float32)
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


def get_df_detector_metrics(
    n_true_bboxes: List[List[Tuple[int, int, int, int]]],
    n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
    minimum_iou: float,
    raw_n_pred_bboxes: List[List[Tuple[int, int, int, int]]] = None,
    raw_n_pred_scores: List[List[float]] = None
) -> pd.DataFrame:
    df_detector_matchings = get_df_detector_matchings(n_true_bboxes,
                                                      n_pred_bboxes,
                                                      minimum_iou)

    detector_TP = df_detector_matchings['TP'].sum()
    detector_FP = df_detector_matchings['FP'].sum()
    detector_FN = df_detector_matchings['FN'].sum()
    detector_precision = detector_TP / (detector_TP + detector_FP)
    detector_recall = detector_TP / (detector_TP + detector_FN)
    detector_fscore = 2 / (1 / detector_precision + 1 / detector_recall)

    df_bboxes = pd.DataFrame([
        item for sublist in df_detector_matchings['items'].tolist()
        for item in sublist
    ])
    iou_mean = np.mean(df_bboxes[df_bboxes['iou'] > 0.]['iou'])

    coco_metrics = count_coco_metrics(n_true_bboxes,
                                      raw_n_pred_bboxes,
                                      raw_n_pred_scores)
    for k in coco_metrics:
        coco_metrics[k] = [round(coco_metrics[k], 3)]
    df_detector_metrics = pd.DataFrame({
        'TP': [detector_TP],
        'FP': [detector_FP],
        'FN': [detector_FN],
        'precision': [round(detector_precision, 3)],
        'recall': [round(detector_recall, 3)],
        'fscore': [round(detector_fscore, 3)],
        'iou_mean': [round(iou_mean, 3)],
        **coco_metrics
    }).T
    df_detector_metrics.columns = ['value']
    df_detector_metrics = df_detector_metrics.astype(object)

    return df_detector_metrics


def get_df_detector_metrics_recall(
    n_true_bboxes: List[List[Tuple[int, int, int, int]]],
    n_pred_bboxes: List[List[Tuple[int, int, int, int]]],
    n_true_labels: List[List[str]],
    minimum_iou: float
) -> pd.DataFrame:

    n_true_bboxes = [
        [tuple(bbox) for bbox in sublist]
        for sublist in n_true_bboxes
    ]
    n_pred_bboxes = [
        [tuple(bbox) for bbox in sublist]
        for sublist in n_pred_bboxes
    ]

    df_bboxes_all = get_df_all_bboxes_matchings(
        n_true_bboxes,
        n_pred_bboxes,
        minimum_iou
    )
    for idx in df_bboxes_all.index:
        img_idx = df_bboxes_all.loc[idx, 'indexes']
        is_true_bbox = df_bboxes_all.loc[idx, 'is_true_bbox']

        if is_true_bbox:
            true_bbox = df_bboxes_all.loc[idx, 'true_bbox']
            img_bbox_idx = n_true_bboxes[img_idx].index(tuple(true_bbox))
            df_bboxes_all.loc[
                idx, 'true_label'
            ] = n_true_labels[img_idx][img_bbox_idx]
        else:
            df_bboxes_all.loc[idx, 'true_label'] = "FP"

    metrics = {}
    all_classes = np.unique(df_bboxes_all['true_label'].tolist())
    all_classes = all_classes[all_classes != "FP"]
    for cl in all_classes:
        df_class = df_bboxes_all.query(f'true_label == "{cl}"')
        df_TP = df_class.query('found')
        df_FN = df_class.query('not found')
        recall = round(len(df_TP) / (len(df_TP) + len(df_FN) + 1e-6), 3)
        metrics[cl] = {
            "support": len(df_class),
            'TP': len(df_TP),
            'FN': len(df_FN),
            "recall": recall,
        }

    df_detector_metrics_recall = pd.DataFrame(metrics, dtype=object).T

    df_detector_metrics_recall_nonempty = df_detector_metrics_recall.query(
        'support > 0'
    )
    macro_average_recall = np.mean(
        df_detector_metrics_recall_nonempty['recall']
    )
    weighted_average_recall = np.average(
        df_detector_metrics_recall_nonempty['recall'],
        weights=df_detector_metrics_recall_nonempty['support']
    )

    support = np.sum(df_detector_metrics_recall['support'])
    TP = np.sum(df_detector_metrics_recall['TP'])
    FN = np.sum(df_detector_metrics_recall['FN'])

    df_detector_metrics_recall = df_detector_metrics_recall.append(
        pd.DataFrame([{
            'support': support,
            'TP': TP,
            'FN': FN,
            'recall': round(macro_average_recall, 3),
        }], index=['macro average'])
    )
    df_detector_metrics_recall = df_detector_metrics_recall.append(
        pd.DataFrame([{
            'support': support,
            'TP': TP,
            'FN': FN,
            'recall': round(weighted_average_recall, 3),
        }], index=['weighted average'])
    )

    df_detector_metrics_recall = df_detector_metrics_recall[
        ['support', 'TP', 'FN', 'recall']
    ]
    df_detector_metrics_recall = df_detector_metrics_recall.sort_values(
        by='support', ascending=False
    )

    return df_detector_metrics_recall
