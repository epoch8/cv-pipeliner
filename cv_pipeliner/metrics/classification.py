from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from cv_pipeliner.core.data import BboxData


def _add_metrics_to_dict(
    classification_metrics: Dict,
    labels: List[str],
    tops_n: List[int],
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
    for top_n in tops_n:
        if top_n == 1:
            continue
        TP_top_n = np.sum([classification_metrics[class_name][f'TP@{top_n}'] for class_name in labels])
        FP_top_n = np.sum([classification_metrics[class_name][f'FP@{top_n}'] for class_name in labels])
        precisions_top_n = [classification_metrics[class_name][f'precision@{top_n}'] for class_name in labels]
        micro_average_precision_top_n = TP_top_n / max(TP_top_n + FP_top_n, 1e-6)
        macro_average_precision_top_n = np.average(precisions_top_n)
        weighted_average_precision_top_n = np.average(precisions_top_n, weights=supports)
        for average in ['micro_average', 'macro_average', 'weighted_average']:
            classification_metrics[f'{prefix_caption}{average}{postfix_caption}'][f'TP@{top_n}'] = (
                TP_top_n
            )
            classification_metrics[f'{prefix_caption}{average}{postfix_caption}'][f'FP@{top_n}'] = (
                FP_top_n
            )
        classification_metrics[f'{prefix_caption}micro_average{postfix_caption}'][f'precision@{top_n}'] = (
            micro_average_precision_top_n
        )
        classification_metrics[f'{prefix_caption}macro_average{postfix_caption}'][f'precision@{top_n}'] = (
            macro_average_precision_top_n
        )
        classification_metrics[f'{prefix_caption}weighted_average{postfix_caption}'][f'precision@{top_n}'] = (
            weighted_average_precision_top_n
        )


def get_TP_and_FP_top_n(
    true_labels: List[str],
    pred_labels_top_n: List[List[str]],
) -> Tuple[int, int]:
    TP, FP = 0, 0
    for true_label, pred_label_top_n in zip(true_labels, pred_labels_top_n):
        if true_label in pred_label_top_n:
            TP += 1
        else:
            FP += 1
    return TP, FP


def get_df_classification_metrics(
    n_true_bboxes_data: List[List[BboxData]],
    n_pred_bboxes_data: List[List[BboxData]],
    pseudo_class_names: List[str],
    known_class_names: List[str] = None,
    tops_n: List[int] = [1]
) -> pd.DataFrame:
    # We use pipeline metrics for it:

    assert len(n_true_bboxes_data) == len(n_true_bboxes_data)
    true_bboxes_data = np.array([bbox_data for bboxes_data in n_true_bboxes_data for bbox_data in bboxes_data])
    pred_bboxes_data = np.array([bbox_data for bboxes_data in n_pred_bboxes_data for bbox_data in bboxes_data])
    assert len(true_bboxes_data) == len(pred_bboxes_data)
    true_labels = np.array([bbox_data.label for bbox_data in true_bboxes_data])
    pred_labels = np.array([bbox_data.label for bbox_data in pred_bboxes_data])
    pred_labels_top_n = np.array([
        [top_n_label for top_n_label in bbox_data.labels_top_n]
        for bbox_data in pred_bboxes_data

    ])
    assert max(tops_n) <= min([bbox_data.top_n for bbox_data in pred_bboxes_data])

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
        if max(tops_n) > 1:
            for top_n in tops_n:
                if top_n == 1:
                    continue
                TP_by_class_name_top_n, FP_by_class_name_top_n = get_TP_and_FP_top_n(
                    true_labels=true_labels[true_labels == class_name],
                    pred_labels_top_n=pred_labels_top_n[true_labels == class_name]
                )
                classification_metrics[class_name][f'TP@{top_n}'] = TP_by_class_name_top_n
                classification_metrics[class_name][f'FP@{top_n}'] = FP_by_class_name_top_n
                classification_metrics[class_name][f'precision@{top_n}'] = TP_by_class_name_top_n / max(
                    TP_by_class_name_top_n + FP_by_class_name_top_n, 1e-6
                )
    _add_metrics_to_dict(
        classification_metrics=classification_metrics,
        labels=all_class_names,
        tops_n=tops_n,
        prefix_caption='all_'
    )
    _add_metrics_to_dict(
        classification_metrics=classification_metrics,
        labels=class_names_without_pseudo_classes,
        tops_n=tops_n,
        prefix_caption='all_',
        postfix_caption='_without_pseudo_classes'
    )
    if known_class_names is not None:
        known_class_names = list(set(all_class_names).intersection(set(known_class_names)))
        known_class_names_without_pseudo_classes = list(set(known_class_names) - set(pseudo_class_names))
        _add_metrics_to_dict(
            classification_metrics=classification_metrics,
            labels=known_class_names,
            tops_n=tops_n,
            prefix_caption='known_'
        )
        _add_metrics_to_dict(
            classification_metrics=classification_metrics,
            labels=known_class_names_without_pseudo_classes,
            tops_n=tops_n,
            prefix_caption='known_',
            postfix_caption='_without_pseudo_classes'
        )

    df_classification_metrics = pd.DataFrame(classification_metrics, dtype=object).T
    df_classification_metrics.sort_values(by='support', ascending=False, inplace=True)
    df_classification_metrics_columns = ['support', 'recall', 'f1_score', 'value'] + [
        f'precision@{top_n}' for top_n in tops_n if top_n > 1
    ] + ['precision', 'recall', 'f1_score', 'value', 'TP', 'FP', 'FN'] + [
        item for sublist in [[f'TP@{top_n}', f'FP@{top_n}'] for top_n in tops_n if top_n > 1]
        for item in sublist
    ]
    df_classification_metrics = df_classification_metrics[df_classification_metrics_columns]

    if known_class_names is not None:
        df_classification_metrics.loc[all_class_names, 'known'] = (
            df_classification_metrics.loc[all_class_names].index.isin(known_class_names)
        )
        df_classification_metrics.loc[all_class_names, 'pseudo'] = (
            df_classification_metrics.loc[all_class_names].index.isin(pseudo_class_names)
        )

    return df_classification_metrics
