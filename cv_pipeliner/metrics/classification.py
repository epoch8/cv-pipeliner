from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from cv_pipeliner.core.data import BboxData


def _add_metrics_to_dict(
    classification_metrics: Dict,
    labels: List[str],
    prefix_caption: str = '',
    postfix_caption: str = '',
):
    supports = [classification_metrics[class_name]['support'] for class_name in labels]
    support = np.sum(supports)
    TP = np.sum([classification_metrics[class_name]['TP'] for class_name in labels])
    FP = np.sum([classification_metrics[class_name]['FP'] for class_name in labels])
    FN = np.sum([classification_metrics[class_name]['FN'] for class_name in labels])
    accuracy = TP / max(TP + FP + FN, 1e-6)
    micro_average_precision = TP / max(TP + FP, 1e-6)
    micro_average_recall = TP / max(TP + FN, 1e-6)
    micro_average_f1_score = 2 * micro_average_precision * micro_average_recall / (
        max(micro_average_precision + micro_average_recall, 1e-6)
    )
    precisions = [classification_metrics[class_name]['precision'] for class_name in labels]
    recalls = [classification_metrics[class_name]['recall'] for class_name in labels]
    f1_scores = [classification_metrics[class_name]['f1_score'] for class_name in labels]
    macro_average_precision = np.average(precisions)
    weighted_average_precision = np.average(precisions, weights=supports)
    macro_average_recall = np.average(recalls)
    weighted_average_recall = np.average(recalls, weights=supports)
    macro_average_f1_score = np.average(f1_scores)
    weighted_average_f1_score = np.average(f1_scores, weights=supports)
    sum_support = np.sum(supports)
    classification_metrics[f'{prefix_caption}accuracy{postfix_caption}'] = {
        'support': support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'value': accuracy
    }
    classification_metrics[f'{prefix_caption}micro_average{postfix_caption}'] = {
        'support': sum_support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': micro_average_precision,
        'recall': micro_average_recall,
        'f1_score': micro_average_f1_score
    }
    classification_metrics[f'{prefix_caption}macro_average{postfix_caption}'] = {
        'support': sum_support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': macro_average_precision,
        'recall': macro_average_recall,
        'f1_score': macro_average_f1_score
    }
    classification_metrics[f'{prefix_caption}weighted_average{postfix_caption}'] = {
        'support': sum_support,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': weighted_average_precision,
        'recall': weighted_average_recall,
        'f1_score': weighted_average_f1_score
    }


def get_df_classification_metrics(
    n_true_bboxes_data: List[List[BboxData]],
    n_pred_bboxes_data: List[List[BboxData]],
    pseudo_class_names: List[str] = ['trash', 'not_part', 'other'],
    known_class_names: List[str] = None
) -> pd.DataFrame:
    # We use pipeline metrics for it:

    assert len(n_true_bboxes_data) == len(n_true_bboxes_data)
    true_bboxes_data = np.array([bbox_data for bboxes_data in n_true_bboxes_data for bbox_data in bboxes_data])
    pred_bboxes_data = np.array([bbox_data for bboxes_data in n_pred_bboxes_data for bbox_data in bboxes_data])
    assert len(true_bboxes_data) == len(pred_bboxes_data)
    true_labels = np.array([bbox_data.label for bbox_data in true_bboxes_data])
    pred_labels = np.array([bbox_data.label for bbox_data in pred_bboxes_data])

    all_class_names = np.unique(np.concatenate([true_labels, pred_labels])).tolist()
    class_names_without_pseudo_classes = list(set(all_class_names) - set(pseudo_class_names))
    MMM = multilabel_confusion_matrix(
        y_true=true_labels,
        y_pred=pred_labels,
        labels=all_class_names
    )
    classification_metrics = {}
    for idx, class_name in enumerate(all_class_names):
        support_by_class_name = np.sum(true_labels == class_name)
        TP_by_class_name = MMM[idx, 1, 1]
        FP_by_class_name = MMM[idx, 0, 1]
        FN_by_class_name = MMM[idx, 1, 0]
        precision_by_class_name = TP_by_class_name / max(TP_by_class_name + FP_by_class_name, 1e-6)
        recall_by_class_name = TP_by_class_name / max(TP_by_class_name + FN_by_class_name, 1e-6)
        f1_score_by_class_name = 2 * precision_by_class_name * recall_by_class_name / (
            max(precision_by_class_name + recall_by_class_name, 1e-6)
        )
        classification_metrics[class_name] = {
            'support': support_by_class_name,
            'TP': TP_by_class_name,
            'FP': FP_by_class_name,
            'FN': FN_by_class_name,
            'precision': precision_by_class_name,
            'recall': recall_by_class_name,
            'f1_score': f1_score_by_class_name
        }
    _add_metrics_to_dict(
        classification_metrics=classification_metrics,
        labels=all_class_names,
        prefix_caption='all_'
    )
    _add_metrics_to_dict(
        classification_metrics=classification_metrics,
        labels=class_names_without_pseudo_classes,
        prefix_caption='all_',
        postfix_caption='_without_pseudo_classes'
    )
    if known_class_names is not None:
        known_class_names_without_pseudo_classes = list(set(known_class_names) - set(pseudo_class_names))
        _add_metrics_to_dict(
            classification_metrics=classification_metrics,
            labels=known_class_names,
            prefix_caption='known_'
        )
        _add_metrics_to_dict(
            classification_metrics=classification_metrics,
            labels=known_class_names_without_pseudo_classes,
            prefix_caption='known_',
            postfix_caption='_without_pseudo_classes'
        )

    df_classification_metrics = pd.DataFrame(classification_metrics, dtype=object).T
    df_classification_metrics.sort_values(by='support', ascending=False, inplace=True)
    df_classification_metrics = df_classification_metrics[
        ['support', 'value', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1_score']
    ]

    if known_class_names is not None:
        df_classification_metrics.loc[all_class_names, 'known'] = (
            df_classification_metrics.loc[all_class_names].index.isin(known_class_names)
        )
        df_classification_metrics.loc[all_class_names, 'pseudo'] = (
            df_classification_metrics.loc[all_class_names].index.isin(pseudo_class_names)
        )

    return df_classification_metrics
