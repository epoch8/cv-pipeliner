from typing import Dict, List

import pandas as pd
import numpy as np

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.metrics.image_data_matching import BboxDataMatching, ImageDataMatching


def _count_errors_types_and_get_pipeline_metrics_per_class(
    images_data_matchings: List[ImageDataMatching],
    labels: List[str],
    extra_bbox_label: str,
    filter_by_true_labels: bool
) -> Dict:
    pipeline_metrics_per_class = {}
    if filter_by_true_labels:
        images_data_matchings = [
            ImageDataMatching(
                true_image_data=image_data_matching.true_image_data,
                pred_image_data=image_data_matching.pred_image_data,
                minimum_iou=image_data_matching.minimum_iou,
                extra_bbox_label=image_data_matching.extra_bbox_label,
                bboxes_data_matchings=[
                    BboxDataMatching(
                        true_bbox_data=bbox_data_matching.true_bbox_data,
                        pred_bbox_data=bbox_data_matching.pred_bbox_data,
                        extra_bbox_label=bbox_data_matching.extra_bbox_label
                    )
                    for bbox_data_matching in image_data_matching.bboxes_data_matchings
                    if (
                        bbox_data_matching.true_bbox_data is not None
                        and bbox_data_matching.true_bbox_data.label in labels
                    )
                ]
            )
            for image_data_matching in images_data_matchings
        ]
    true_labels = np.array([
        bbox_data_matching.true_bbox_data.label
        for image_data_matching in images_data_matchings
        for bbox_data_matching in image_data_matching.bboxes_data_matchings
        if bbox_data_matching.true_bbox_data is not None
    ])
    for class_name in labels:
        support_by_class_name = np.sum(true_labels == class_name)
        TP_by_class_name = np.sum([
            image_data_matching.get_pipeline_TP(label=class_name)
            for image_data_matching in images_data_matchings
        ])
        FP_by_class_name = np.sum([
            image_data_matching.get_pipeline_FP(label=class_name)
            for image_data_matching in images_data_matchings
        ])
        if class_name != extra_bbox_label:
            TP_extra_bbox_by_class_name = np.sum([
                image_data_matching.get_pipeline_TP_extra_bbox(label=class_name)
                for image_data_matching in images_data_matchings
            ])
            FP_extra_bbox_by_class_name = np.sum([
                image_data_matching.get_pipeline_FP_extra_bbox(label=class_name)
                for image_data_matching in images_data_matchings
            ])
            FN_extra_bbox_by_class_name = np.sum([
                image_data_matching.get_pipeline_FN_extra_bbox(label=class_name)
                for image_data_matching in images_data_matchings
            ])
        else:
            TP_extra_bbox_by_class_name = None
            FP_extra_bbox_by_class_name = None
            FN_extra_bbox_by_class_name = None
        FN_by_class_name = np.sum([
            image_data_matching.get_pipeline_FN(label=class_name)
            for image_data_matching in images_data_matchings
        ])
        TP_extra_bbox_in_precision_numerator = (
            0 if TP_extra_bbox_by_class_name is None else TP_extra_bbox_by_class_name
        )
        FP_extra_bbox_in_precision_denominator = (
            0 if FP_extra_bbox_by_class_name is None else FP_extra_bbox_by_class_name
        )
        FN_extra_bbox_in_precision_denominator = (
            0 if FN_extra_bbox_by_class_name is None else FN_extra_bbox_by_class_name
        )
        precision_by_class_name = (TP_by_class_name + TP_extra_bbox_in_precision_numerator) / max(
            TP_by_class_name + FP_by_class_name + FP_extra_bbox_in_precision_denominator, 1e-6
        )
        recall_by_class_name = (TP_by_class_name + TP_extra_bbox_in_precision_numerator) / max(
            TP_by_class_name + FN_by_class_name + FN_extra_bbox_in_precision_denominator, 1e-6
        )
        f1_score_by_class_name = 2 * precision_by_class_name * recall_by_class_name / (
            max(precision_by_class_name + recall_by_class_name, 1e-6)
        )
        ious = [
            bbox_data_matching.iou
            for image_data_matching in images_data_matchings
            for bbox_data_matching in image_data_matching.bboxes_data_matchings
            if bbox_data_matching.iou is not None and bbox_data_matching.true_bbox_data is not None and
            bbox_data_matching.true_bbox_data.label == class_name
        ]
        iou_mean = np.mean(ious) if len(ious) > 0 else 0
        pipeline_metrics_per_class[class_name] = {
            'support': support_by_class_name,
            'TP': TP_by_class_name,
            'FP': FP_by_class_name,
            'FN': FN_by_class_name,
            'TP (extra bbox)': TP_extra_bbox_by_class_name,
            'FP (extra bbox)': FP_extra_bbox_by_class_name,
            'FN (extra bbox)': FN_extra_bbox_by_class_name,
            'iou_mean': iou_mean,
            'precision': precision_by_class_name,
            'recall': recall_by_class_name,
            'f1_score': f1_score_by_class_name
        }
    return pipeline_metrics_per_class


def _add_metrics_to_dict(
    pipeline_metrics_per_class: Dict,
    pipeline_metrics: Dict,
    labels: List[str],
    prefix_caption: str = '',
    postfix_caption: str = '',
):
    supports = [pipeline_metrics_per_class[class_name]['support'] for class_name in labels]
    support = np.sum(supports)
    TP = np.sum([pipeline_metrics_per_class[class_name]['TP'] for class_name in labels])
    FP = np.sum([pipeline_metrics_per_class[class_name]['FP'] for class_name in labels])
    FN = np.sum([pipeline_metrics_per_class[class_name]['FN'] for class_name in labels])
    TP_extra_bbox = np.sum([
        pipeline_metrics_per_class[class_name]['TP (extra bbox)'] for class_name in labels
        if pipeline_metrics_per_class[class_name]['TP (extra bbox)'] is not None
    ])
    FP_extra_bbox = np.sum([
        pipeline_metrics_per_class[class_name]['FP (extra bbox)'] for class_name in labels
        if pipeline_metrics_per_class[class_name]['FP (extra bbox)'] is not None
    ])
    FN_extra_bbox = np.sum([
        pipeline_metrics_per_class[class_name]['FN (extra bbox)'] for class_name in labels
        if pipeline_metrics_per_class[class_name]['FN (extra bbox)'] is not None
    ])
    iou_mean = np.mean([
        pipeline_metrics_per_class[class_name]['iou_mean'] for class_name in labels
        if pipeline_metrics_per_class[class_name]['iou_mean'] is not None
    ])
    accuracy = (TP + TP_extra_bbox) / max(TP + FN + FN + TP_extra_bbox + FP_extra_bbox, 1e-6)
    micro_average_precision = (TP + TP_extra_bbox) / max(TP + FP + FP_extra_bbox, 1e-6)
    micro_average_recall = TP / max(TP + FN, 1e-6)
    micro_average_f1_score = 2 * micro_average_precision * micro_average_recall / (
        max(micro_average_precision + micro_average_recall, 1e-6)
    )
    precisions = [pipeline_metrics_per_class[class_name]['precision'] for class_name in labels]
    recalls = [pipeline_metrics_per_class[class_name]['recall'] for class_name in labels]
    f1_scores = [pipeline_metrics_per_class[class_name]['f1_score'] for class_name in labels]
    macro_average_precision = np.average(precisions)
    weighted_average_precision = np.average(precisions, weights=supports)
    macro_average_recall = np.average(recalls)
    weighted_average_recall = np.average(recalls, weights=supports)
    macro_average_f1_score = np.average(f1_scores)
    weighted_average_f1_score = np.average(f1_scores, weights=supports)
    sum_support = np.sum(supports)
    pipeline_metrics[f'{prefix_caption}accuracy{postfix_caption}'] = {
        'support': support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TP (extra bbox)': TP_extra_bbox,
        'FP (extra bbox)': FP_extra_bbox,
        'FN (extra bbox)': FN_extra_bbox,
        'value': accuracy
    }
    pipeline_metrics[f'{prefix_caption}iou_mean{postfix_caption}'] = {
        'support': support,
        'value': iou_mean
    }
    pipeline_metrics[f'{prefix_caption}micro_average{postfix_caption}'] = {
        'support': sum_support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TP (extra bbox)': TP_extra_bbox,
        'FP (extra bbox)': FP_extra_bbox,
        'FN (extra bbox)': FN_extra_bbox,
        'precision': micro_average_precision,
        'recall': micro_average_recall,
        'f1_score': micro_average_f1_score
    }
    pipeline_metrics[f'{prefix_caption}macro_average{postfix_caption}'] = {
        'support': sum_support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TP (extra bbox)': TP_extra_bbox,
        'FP (extra bbox)': FP_extra_bbox,
        'FN (extra bbox)': FN_extra_bbox,
        'precision': macro_average_precision,
        'recall': macro_average_recall,
        'f1_score': macro_average_f1_score
    }
    pipeline_metrics[f'{prefix_caption}weighted_average{postfix_caption}'] = {
        'support': sum_support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TP (extra bbox)': TP_extra_bbox,
        'FP (extra bbox)': FP_extra_bbox,
        'FN (extra bbox)': FN_extra_bbox,
        'precision': weighted_average_precision,
        'recall': weighted_average_recall,
        'f1_score': weighted_average_f1_score
    }

from tqdm import tqdm
def get_df_pipeline_metrics(
    true_images_data: List[ImageData],
    pred_images_data: List[ImageData],
    minimum_iou: float,
    extra_bbox_label: str,
    pseudo_class_names: List[str] = ['trash', 'not_part', 'other'],
    known_class_names: List[str] = None
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
        )
        for true_image_data, pred_image_data in tqdm(list(zip(true_images_data, pred_images_data)))
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
    all_class_names = np.unique(np.concatenate([true_labels, pred_labels]))
    pipeline_metrics_per_class_all_class_names = _count_errors_types_and_get_pipeline_metrics_per_class(
        images_data_matchings=images_data_matchings,
        labels=all_class_names,
        extra_bbox_label=extra_bbox_label,
        filter_by_true_labels=False
    )
    pipeline_metrics = {}
    for class_name in all_class_names:
        class_name_caption = f"{class_name} (pseudo-class)" if class_name in pseudo_class_names else class_name
        pipeline_metrics[class_name_caption] = pipeline_metrics_per_class_all_class_names[class_name]
    TP_extra_bbox = np.sum([
        image_data_matching.get_pipeline_TP_extra_bbox(label=extra_bbox_label)
        for image_data_matching in images_data_matchings
    ])
    FP_extra_bbox = np.sum([
        image_data_matching.get_pipeline_FP_extra_bbox(label=extra_bbox_label)
        for image_data_matching in images_data_matchings
    ])
    FN_extra_bbox = np.sum([
        image_data_matching.get_pipeline_FN_extra_bbox(label=extra_bbox_label)
        for image_data_matching in images_data_matchings
    ])
    precision_extra_bbox = TP_extra_bbox / max(TP_extra_bbox + FP_extra_bbox, 1e-6)
    recall_extra_bbox = TP_extra_bbox / max(TP_extra_bbox + FN_extra_bbox, 1e-6)
    f1_score_extra_bbox = 2 * precision_extra_bbox * recall_extra_bbox / (
        max(precision_extra_bbox + recall_extra_bbox, 1e-6)
    )
    extra_bbox_label_caption = f'{extra_bbox_label} (extra bbox)' if extra_bbox_label is not None else 'extra bbox'
    pipeline_metrics[extra_bbox_label_caption] = {
        'support': TP_extra_bbox + FP_extra_bbox + FN_extra_bbox,
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TP (extra bbox)': TP_extra_bbox,
        'FP (extra bbox)': FP_extra_bbox,
        'FN (extra bbox)': FN_extra_bbox,
        'precision': precision_extra_bbox,
        'recall': recall_extra_bbox,
        'f1_score': f1_score_extra_bbox
    }

    class_names_without_pseudo_classes = list(set(all_class_names) - set(pseudo_class_names))
    _add_metrics_to_dict(
        pipeline_metrics_per_class=pipeline_metrics_per_class_all_class_names,
        pipeline_metrics=pipeline_metrics,
        labels=all_class_names,
        prefix_caption='all_'
    )
    _add_metrics_to_dict(
        pipeline_metrics_per_class=pipeline_metrics_per_class_all_class_names,
        pipeline_metrics=pipeline_metrics,
        labels=class_names_without_pseudo_classes,
        postfix_caption='_without_pseudo_classes'
    )
    if known_class_names is not None:
        known_class_names_without_pseudo_classes = list(set(known_class_names) - set(pseudo_class_names))
        pipeline_metrics_per_known_class = _count_errors_types_and_get_pipeline_metrics_per_class(
            images_data_matchings=images_data_matchings,
            labels=known_class_names,
            extra_bbox_label=extra_bbox_label,
            filter_by_true_labels=True
        )
        pipeline_metrics_per_known_class_without_pseudo_classes = _count_errors_types_and_get_pipeline_metrics_per_class(  # noqa: E501
            images_data_matchings=images_data_matchings,
            labels=known_class_names_without_pseudo_classes,
            extra_bbox_label=extra_bbox_label,
            filter_by_true_labels=True
        )
        _add_metrics_to_dict(
            pipeline_metrics_per_class=pipeline_metrics_per_known_class,
            pipeline_metrics=pipeline_metrics,
            labels=known_class_names,
            prefix_caption='known_'
        )
        _add_metrics_to_dict(
            pipeline_metrics_per_class=pipeline_metrics_per_known_class_without_pseudo_classes,
            pipeline_metrics=pipeline_metrics,
            labels=known_class_names_without_pseudo_classes,
            prefix_caption='known_',
            postfix_caption='_without_pseudo_classes'
        )

    df_pipeline_metrics = pd.DataFrame(pipeline_metrics, dtype=object).T
    df_pipeline_metrics = df_pipeline_metrics[
        ['support', 'value', 'TP', 'FP', 'FN', 'TP (extra bbox)', 'FP (extra bbox)', 'FN (extra bbox)', 'precision', 'recall', 'f1_score']
    ]
    df_pipeline_metrics.sort_values(by='support', ascending=False, inplace=True)
    if known_class_names is not None:
        df_pipeline_metrics.loc[class_names_without_pseudo_classes, 'is known by classifier'] = (
            df_pipeline_metrics.loc[class_names_without_pseudo_classes].index.isin(known_class_names)
        )
        pseudo_class_names_exists = [
            class_name
            for class_name in pseudo_class_names
            if f"{class_name} (pseudo-class)" in pipeline_metrics
        ]
        pseudo_class_names_captions = [
            f"{class_name} (pseudo-class)" for class_name in pseudo_class_names_exists
        ]
        df_pipeline_metrics.loc[pseudo_class_names_captions, 'is known by classifier'] = [
            class_name in known_class_names for class_name in pseudo_class_names_exists
        ]

    return df_pipeline_metrics
