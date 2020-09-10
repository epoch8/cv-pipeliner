from typing import List

import pandas as pd
import numpy as np

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.metrics.image_data_matching import ImageDataMatching


df_pipeline_metrics_columns = [
    'support', 'value', 'TP', 'FP', 'FN', 'TP (extra bbox)', 'FP (extra bbox)',
    'classification errors on true bboxes', 'precision', 'recall', 'f1_score'
]


def get_df_pipeline_metrics(
    true_images_data: List[ImageData],
    pred_images_data: List[ImageData],
    minimum_iou: float,
    extra_bbox_label: str = None,
    use_soft_metrics_with_known_labels: List[str] = None,
) -> pd.DataFrame:
    '''
    Returns pipdline metrics (accuracy, precision, recall, f1_score), including metrics per class..
    '''
    images_data_matchings = [
        ImageDataMatching(
            true_image_data=true_image_data,
            pred_image_data=pred_image_data,
            minimum_iou=minimum_iou,
            extra_bbox_label=extra_bbox_label,
            use_soft_metrics_with_known_labels=use_soft_metrics_with_known_labels
        )
        for true_image_data, pred_image_data in zip(true_images_data, pred_images_data)
    ]
    true_labels = np.array([
        bbox_data.label
        for image_data in true_images_data
        for bbox_data in image_data.bboxes_data
    ])
    pred_labels = np.array([
        bbox_data.label
        for image_data in pred_images_data
        for bbox_data in image_data.bboxes_data
    ])
    class_names = np.unique(np.concatenate([true_labels, pred_labels]))
    pipeline_metrics = {}
    for class_name in class_names:
        support_by_class_name = np.sum(true_labels == class_name)
        TP_by_class_name = np.sum([
            image_data_matching.get_pipeline_TP(filter_by_label=class_name)
            for image_data_matching in images_data_matchings
        ])
        FP_by_class_name = np.sum(
            image_data_matching.get_pipeline_FP(filter_by_label=class_name)
            for image_data_matching in images_data_matchings
        )
        if class_name != extra_bbox_label:
            TP_extra_bbox_by_class_name = np.sum([
                image_data_matching.get_pipeline_TP_extra_bbox(filter_by_label=class_name)
                for image_data_matching in images_data_matchings
            ])
            FP_extra_bbox_by_class_name = np.sum([
                image_data_matching.get_pipeline_FP_extra_bbox(filter_by_label=class_name)
                for image_data_matching in images_data_matchings
            ])
        else:
            TP_extra_bbox_by_class_name = None
            FP_extra_bbox_by_class_name = None
        FN_by_class_name = np.sum([
            image_data_matching.get_pipeline_FN(filter_by_label=class_name)
            for image_data_matching in images_data_matchings
        ])
        TP_extra_bbox_in_precision_numerator = 0 if TP_extra_bbox_by_class_name is None else TP_extra_bbox_by_class_name
        FP_extra_bbox_in_precision_denominator = 0 if FP_extra_bbox_by_class_name is None else FP_extra_bbox_by_class_name
        precision_by_class_name = (TP_by_class_name + TP_extra_bbox_in_precision_numerator) / max(
            TP_by_class_name + FP_by_class_name + FP_extra_bbox_in_precision_denominator, 1e-6
        )
        recall_by_class_name = TP_by_class_name / max(TP_by_class_name + FN_by_class_name, 1e-6)
        f1_score_by_class_name = 2 * precision_by_class_name * recall_by_class_name / (
            max(precision_by_class_name + recall_by_class_name, 1e-6)
        )
        classification_errors_count_on_true_bboxes_by_class_name = np.sum(
            image_data_matching.get_classification_errors_count_on_true_bboxes(filter_by_label=class_name)
            for image_data_matching in images_data_matchings
        )
        pipeline_metrics[class_name] = {
            'support': support_by_class_name,
            'TP': TP_by_class_name,
            'FP': FP_by_class_name,
            'FN': FN_by_class_name,
            'TP (extra bbox)': TP_extra_bbox_by_class_name,
            'FP (extra bbox)': FP_extra_bbox_by_class_name,
            'classification errors on true bboxes': classification_errors_count_on_true_bboxes_by_class_name,
            'precision': precision_by_class_name,
            'recall': recall_by_class_name,
            'f1_score': f1_score_by_class_name
        }
    TP_extra_bbox = np.sum([image_data_matching.get_pipeline_TP_extra_bbox() for image_data_matching in images_data_matchings])
    FP_extra_bbox = np.sum([image_data_matching.get_pipeline_FP_extra_bbox() for image_data_matching in images_data_matchings])
    precision_extra_bbox = TP_extra_bbox / max(TP_extra_bbox + FP_extra_bbox, 1e-6)
    if extra_bbox_label is None:
        extra_bbox_label_caption = 'extra bbox'
    else:
        extra_bbox_label_caption = f'{extra_bbox_label} (extra bbox)'
    pipeline_metrics[extra_bbox_label_caption] = {
        'support': TP_extra_bbox+FP_extra_bbox,
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TP (extra bbox)': TP_extra_bbox,
        'FP (extra bbox)': FP_extra_bbox,
        'precision': precision_extra_bbox,
        'recall': None,
        'f1_score': None
    }

    supports = [pipeline_metrics[class_name]['support'] for class_name in class_names]
    TP = np.sum([image_data_matching.get_pipeline_TP() for image_data_matching in images_data_matchings])
    FP = np.sum([image_data_matching.get_pipeline_FP() for image_data_matching in images_data_matchings])
    FN = np.sum([image_data_matching.get_pipeline_FN() for image_data_matching in images_data_matchings])
    iou_mean = np.mean([
        bbox_data_matching.iou
        for image_data_matching in images_data_matchings
        for bbox_data_matching in image_data_matching.bboxes_data_matchings
        if bbox_data_matching.iou is not None
    ])
    accuracy = TP / max(TP + FN + FN + TP_extra_bbox + FP_extra_bbox, 1e-6)
    micro_average_precision = (TP + TP_extra_bbox) / max(TP + FP + FP_extra_bbox, 1e-6)
    micro_average_recall = TP / max(TP + FN, 1e-6)
    micro_average_f1_score = 2 * micro_average_precision * micro_average_recall / (
        max(micro_average_precision + micro_average_recall, 1e-6)
    )
    precisions = [pipeline_metrics[class_name]['precision'] for class_name in class_names]
    recalls = [pipeline_metrics[class_name]['recall'] for class_name in class_names]
    f1_scores = [pipeline_metrics[class_name]['f1_score'] for class_name in class_names]
    macro_average_precision = np.average(precisions)
    weighted_average_precision = np.average(precisions, weights=supports)
    macro_average_recall = np.average(recalls)
    weighted_average_recall = np.average(recalls, weights=supports)
    macro_average_f1_score = np.average(f1_scores)
    weighted_average_f1_score = np.average(f1_scores, weights=supports)
    sum_support = np.sum(supports)
    classification_errors_count_on_true_bboxes = np.sum(
        [[image_data_matching.get_classification_errors_count_on_true_bboxes(filter_by_label=class_name)
          for image_data_matching in images_data_matchings]
         for class_name in class_names]
    )
    pipeline_metrics['accuracy'] = {
        'support': sum_support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TP (extra bbox)': TP_extra_bbox,
        'FP (extra bbox)': FP_extra_bbox,
        'classification errors on true bboxes': classification_errors_count_on_true_bboxes,
        'value': accuracy
    }
    pipeline_metrics['iou_mean'] = {
        'support': sum_support,
        'value': iou_mean
    }
    pipeline_metrics['micro_average'] = {
        'support': sum_support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TP (extra bbox)': TP_extra_bbox,
        'FP (extra bbox)': FP_extra_bbox,
        'classification errors on true bboxes': classification_errors_count_on_true_bboxes,
        'precision': micro_average_precision,
        'recall': micro_average_recall,
        'f1_score': micro_average_f1_score
    }
    pipeline_metrics['macro_average'] = {
        'support': sum_support,
        'precision': macro_average_precision,
        'recall': macro_average_recall,
        'f1_score': macro_average_f1_score
    }
    pipeline_metrics['weighted_average'] = {
        'support': sum_support,
        'precision': weighted_average_precision,
        'recall': weighted_average_recall,
        'f1_score': weighted_average_f1_score
    }
    if use_soft_metrics_with_known_labels is not None:
        known_supports = [
            pipeline_metrics[class_name]['support'] for class_name in use_soft_metrics_with_known_labels
            if class_name in pipeline_metrics
        ]
        sum_known_supports = np.sum(known_supports)
        known_TP = np.sum(
            [[image_data_matching.get_classification_correct_count_on_true_bboxes(filter_by_label=class_name)
              for image_data_matching in images_data_matchings]
             for class_name in use_soft_metrics_with_known_labels]
        )
        known_FP = np.sum(
            [[image_data_matching.get_classification_errors_count_on_true_bboxes(filter_by_label=class_name)
              for image_data_matching in images_data_matchings]
             for class_name in use_soft_metrics_with_known_labels]
        )
        known_FN = np.sum(
            [[image_data_matching.get_pipeline_FN(filter_by_label=class_name)
              for image_data_matching in images_data_matchings]
             for class_name in use_soft_metrics_with_known_labels]
        )
        known_TP_extra_bbox = np.sum(
            [[image_data_matching.get_pipeline_TP_extra_bbox(filter_by_label=class_name)
              for image_data_matching in images_data_matchings]
             for class_name in use_soft_metrics_with_known_labels]
        )
        known_FP_extra_bbox = np.sum(
            [[image_data_matching.get_pipeline_FP_extra_bbox(filter_by_label=class_name)
              for image_data_matching in images_data_matchings]
             for class_name in use_soft_metrics_with_known_labels]
        )
        known_accuracy = known_TP / max(known_TP + known_FN + known_FN + known_TP_extra_bbox + known_FP_extra_bbox, 1e-6)
        known_micro_average_precision = (known_TP + known_TP_extra_bbox) / max(known_TP + known_FP + known_FP_extra_bbox, 1e-6)
        known_micro_average_recall = known_TP / max(known_TP + known_FN, 1e-6)
        known_micro_average_f1_score = 2 * known_micro_average_precision * known_micro_average_recall / (
            max(known_micro_average_precision + known_micro_average_recall, 1e-6)
        )
        pipeline_metrics['known_accuracy'] = {
            'support': sum_known_supports,
            'TP': known_TP,
            'FP': known_FP,
            'FN': known_FN,
            'TP (extra bbox)': known_TP_extra_bbox,
            'FP (extra bbox)': known_FP_extra_bbox,
            'classification errors on true bboxes': classification_errors_count_on_true_bboxes,
            'value': known_accuracy
        }
        pipeline_metrics['known_micro_average'] = {
            'support': sum_known_supports,
            'TP': known_TP,
            'FP': known_FP,
            'FN': known_FN,
            'TP (extra bbox)': known_TP_extra_bbox,
            'FP (extra bbox)': known_FP_extra_bbox,
            'precision': known_micro_average_precision,
            'recall': known_micro_average_recall,
            'f1_score': known_micro_average_f1_score
        }
        known_precisions = [
            pipeline_metrics[class_name]['precision'] for class_name in use_soft_metrics_with_known_labels
            if class_name in pipeline_metrics
        ]
        known_recalls = [
            pipeline_metrics[class_name]['recall'] for class_name in use_soft_metrics_with_known_labels
            if class_name in pipeline_metrics
        ]
        known_f1_scores = [
            pipeline_metrics[class_name]['f1_score'] for class_name in use_soft_metrics_with_known_labels
            if class_name in pipeline_metrics
        ]
        known_macro_average_precision = np.average(known_precisions)
        known_weighted_average_precision = np.average(known_precisions, weights=known_supports)
        known_macro_average_recall = np.average(known_recalls)
        known_weighted_average_recall = np.average(known_recalls, weights=known_supports)
        known_macro_average_f1_score = np.average(known_f1_scores)
        known_weighted_average_f1_score = np.average(known_f1_scores, weights=known_supports)
        pipeline_metrics['known_macro_average'] = {
            'support': sum_known_supports,
            'precision': known_macro_average_precision,
            'recall': known_macro_average_recall,
            'f1_score': known_macro_average_f1_score
        }
        pipeline_metrics['known_weighted_average'] = {
            'support': sum_known_supports,
            'precision': known_weighted_average_precision,
            'recall': known_weighted_average_recall,
            'f1_score': known_weighted_average_f1_score
        }

    df_pipeline_metrics = pd.DataFrame(pipeline_metrics, dtype=object).T
    df_pipeline_metrics = df_pipeline_metrics[df_pipeline_metrics_columns]
    df_pipeline_metrics.sort_values(by='support', ascending=False, inplace=True)
    if use_soft_metrics_with_known_labels is not None:
        df_pipeline_metrics.loc[class_names, 'is known by classifier'] = (
            df_pipeline_metrics.loc[class_names].index.isin(use_soft_metrics_with_known_labels)
        )

    return df_pipeline_metrics
