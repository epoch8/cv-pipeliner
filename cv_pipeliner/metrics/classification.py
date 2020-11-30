from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix

from cv_pipeliner.core.data import BboxData


def _add_metrics_to_dict(
    classification_metrics: Dict,
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    prefix_caption: str = '',
    postfix_caption: str = '',
):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    support = len(y_true)
    for average in ['micro', 'macro', 'weighted']:
        average_precision, average_recall, average_f1_score, _ = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average=average,
            labels=labels,
            zero_division=0
        )
        classification_metrics[f'{prefix_caption}{average}_average{postfix_caption}'] = {
            'support': support,
            'precision': average_precision,
            'recall': average_recall,
            'f1_score': average_f1_score
        }
    classification_metrics[f'{prefix_caption}accuracy{postfix_caption}'] = {
        'support': support,
        'value': np.sum(y_true != y_pred) / support
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
        class_name_caption = f"{class_name} (pseudo-class)" if class_name in pseudo_class_names else class_name
        classification_metrics[class_name_caption] = {
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
        y_true=true_labels,
        y_pred=pred_labels,
        labels=all_class_names,
        prefix_caption='all_'
    )
    _add_metrics_to_dict(
        classification_metrics=classification_metrics,
        y_true=true_labels,
        y_pred=pred_labels,
        labels=class_names_without_pseudo_classes,
        postfix_caption='_without_pseudo_classes'
    )
    if known_class_names is not None:
        known_class_names_without_pseudo_classes = list(set(known_class_names) - set(pseudo_class_names))
        known_true_labels_idxs = [label in known_class_names for label in true_labels]
        known_true_labels = true_labels[known_true_labels_idxs]
        known_pred_labels = pred_labels[known_true_labels_idxs]
        known_true_labels_without_pseudo_classes_idxs = [label in known_class_names for label in true_labels]
        known_true_labels_without_pseudo_classes = true_labels[known_true_labels_without_pseudo_classes_idxs]
        known_pred_labels_without_pseudo_classes = pred_labels[known_true_labels_without_pseudo_classes_idxs]
        _add_metrics_to_dict(
            classification_metrics=classification_metrics,
            y_true=known_true_labels,
            y_pred=known_pred_labels,
            labels=known_class_names,
            prefix_caption='known_'
        )
        _add_metrics_to_dict(
            classification_metrics=classification_metrics,
            y_true=known_true_labels_without_pseudo_classes,
            y_pred=known_pred_labels_without_pseudo_classes,
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
        df_classification_metrics.loc[class_names_without_pseudo_classes, 'is known by classifier'] = (
            df_classification_metrics.loc[class_names_without_pseudo_classes].index.isin(known_class_names)
        )
        pseudo_class_names_exists = [
            class_name
            for class_name in pseudo_class_names
            if f"{class_name} (pseudo-class)" in classification_metrics
        ]
        pseudo_class_names_captions = [
            f"{class_name} (pseudo-class)" for class_name in pseudo_class_names_exists
        ]
        df_classification_metrics.loc[pseudo_class_names_captions, 'is known by classifier'] = [
            class_name in known_class_names for class_name in pseudo_class_names_exists
        ]

    return df_classification_metrics
