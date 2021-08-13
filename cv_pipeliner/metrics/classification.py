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
    count_mean_expected_steps: bool = False
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
    if count_mean_expected_steps:
        not_nan_labels = [
            class_name
            for class_name in labels
            if not np.isnan(classification_metrics[class_name]['mean_expected_steps'])
        ]
        mean_expected_steps = [
            classification_metrics[class_name]['mean_expected_steps'] for class_name in not_nan_labels
        ]
        mean_expected_steps_supports = [
            classification_metrics[class_name]['support'] for class_name in not_nan_labels
        ]
        macro_average_mean_expected_steps = np.average(mean_expected_steps)
        weighted_average_mean_expected_steps = np.average(
            mean_expected_steps,
            weights=mean_expected_steps_supports
        )
        classification_metrics[f'{prefix_caption}macro_average{postfix_caption}']['mean_expected_steps'] = (
            macro_average_mean_expected_steps
        )
        classification_metrics[f'{prefix_caption}weighted_average{postfix_caption}']['mean_expected_steps'] = (
            weighted_average_mean_expected_steps
        )

    for top_n in tops_n:
        if top_n == 1:
            continue
        precisions_top_n = [classification_metrics[class_name][f'precision@{top_n}'] for class_name in labels]
        recalls_top_n = [classification_metrics[class_name][f'recall@{top_n}'] for class_name in labels]
        f1_score_top_n = [classification_metrics[class_name][f'f1_score@{top_n}'] for class_name in labels]
        macro_average_precision_top_n = np.average(precisions_top_n)
        weighted_average_precision_top_n = np.average(precisions_top_n, weights=supports)
        macro_average_recall_top_n = np.average(recalls_top_n)
        weighted_average_recall_top_n = np.average(recalls_top_n, weights=supports)
        macro_average_f1_score_top_n = np.average(f1_score_top_n)
        weighted_average_f1_score_top_n = np.average(f1_score_top_n, weights=supports)
        classification_metrics[f'{prefix_caption}macro_average{postfix_caption}'][f'precision@{top_n}'] = (
            macro_average_precision_top_n
        )
        classification_metrics[f'{prefix_caption}weighted_average{postfix_caption}'][f'precision@{top_n}'] = (
            weighted_average_precision_top_n
        )
        classification_metrics[f'{prefix_caption}macro_average{postfix_caption}'][f'recall@{top_n}'] = (
            macro_average_recall_top_n
        )
        classification_metrics[f'{prefix_caption}weighted_average{postfix_caption}'][f'recall@{top_n}'] = (
            weighted_average_recall_top_n
        )
        classification_metrics[f'{prefix_caption}macro_average{postfix_caption}'][f'f1_score@{top_n}'] = (
            macro_average_f1_score_top_n
        )
        classification_metrics[f'{prefix_caption}weighted_average{postfix_caption}'][f'f1_score@{top_n}'] = (
            weighted_average_f1_score_top_n
        )


def get_precision_and_recall_top_n(
    true_labels: List[str],
    pred_labels_top_n: List[List[str]],
    label: str,
    top_n: int
) -> Tuple[int, int]:
    # Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
    # Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
    recommended_relevant_items = 0
    recommended_items = 0
    relevant_items = 0
    for true_label, pred_label_top_n in zip(true_labels, pred_labels_top_n):
        if label in pred_label_top_n[0:top_n]:
            recommended_items += 1
        if true_label == label:
            relevant_items += 1
            if label in pred_label_top_n[0:top_n]:
                recommended_relevant_items += 1

    precision_top_n = recommended_relevant_items / max(recommended_items, 1e-6)
    recall_top_n = recommended_relevant_items / max(relevant_items, 1e-6)

    return precision_top_n, recall_top_n


def get_mean_expected_steps(
    n_true_labels: List[List[str]],
    n_pred_labels_top_n: List[List[List[str]]],
    n_pred_scores_top_n: List[List[List[float]]],
    label: str,
    step_penalty: int
) -> Tuple[int, int]:
    n_steps = []
    for true_labels, pred_labels_top_n, pred_scores_top_n in zip(
        n_true_labels, n_pred_labels_top_n, n_pred_scores_top_n
    ):
        idxs_by_label = (pred_labels_top_n == label)
        pred_scores_top_n_by_label = pred_scores_top_n[idxs_by_label]
        idxs_sorted = np.argsort((-1) * pred_scores_top_n_by_label)
        true_labels_sorted = true_labels[idxs_sorted]
        steps = np.where(label == true_labels_sorted)[0]
        nth_step_penalty = step_penalty if len(np.where(label == true_labels)[0]) > 0 else None
        steps = min(step_penalty, np.min(steps) + 1) if len(steps) > 0 else nth_step_penalty
        if steps is not None:
            n_steps.append(steps)
    mean_expected_steps = np.mean(n_steps) if len(n_steps) > 0 else 0
    return mean_expected_steps


def get_df_classification_metrics(
    n_true_bboxes_data: List[List[BboxData]],
    n_pred_bboxes_data: List[List[BboxData]],
    pseudo_class_names: List[str],
    known_class_names: List[str] = None,
    tops_n: List[int] = [1],
    step_penalty: int = 20
) -> pd.DataFrame:
    # We use pipeline metrics for it:

    assert len(n_true_bboxes_data) == len(n_true_bboxes_data)
    true_bboxes_data = np.array([bbox_data for bboxes_data in n_true_bboxes_data for bbox_data in bboxes_data])
    pred_bboxes_data = np.array([bbox_data for bboxes_data in n_pred_bboxes_data for bbox_data in bboxes_data])
    assert len(true_bboxes_data) == len(pred_bboxes_data)
    true_labels = np.array([bbox_data.label for bbox_data in true_bboxes_data])
    pred_labels = np.array([bbox_data.label for bbox_data in pred_bboxes_data])
    pred_labels_top_n = np.array([bbox_data.labels_top_n for bbox_data in pred_bboxes_data])
    min_tops_n_from_pred_bboxes_data = min([bbox_data.top_n for bbox_data in pred_bboxes_data])
    assert max(tops_n) <= min_tops_n_from_pred_bboxes_data
    n_true_labels = [
        np.array([true_bbox_data.label for true_bbox_data in true_bboxes_data])
        for true_bboxes_data in n_true_bboxes_data
    ]
    n_pred_labels_top_n = [
        np.array([
            pred_bbox_data.labels_top_n
            for pred_bbox_data in pred_bboxes_data
        ])
        for pred_bboxes_data in n_pred_bboxes_data
    ]
    n_pred_scores_top_n = [
        np.array([
            pred_bbox_data.classification_scores_top_n
            for pred_bbox_data in pred_bboxes_data
        ])
        for pred_bboxes_data in n_pred_bboxes_data
    ]

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
                precision_top_n, recall_top_n = get_precision_and_recall_top_n(
                    true_labels=true_labels,
                    pred_labels_top_n=pred_labels_top_n,
                    label=class_name,
                    top_n=top_n
                )
                classification_metrics[class_name][f'precision@{top_n}'] = precision_top_n
                classification_metrics[class_name][f'recall@{top_n}'] = recall_top_n
                classification_metrics[class_name][f'f1_score@{top_n}'] = (
                    2 * precision_top_n * recall_top_n
                ) / max(precision_top_n + recall_top_n, 1e-6)

    if known_class_names is not None:
        len_known_class_names = len(known_class_names)
        count_mean_expected_steps = min_tops_n_from_pred_bboxes_data >= len_known_class_names
        known_class_names = list(set(all_class_names).intersection(set(known_class_names)))
        known_class_names_without_pseudo_classes = list(set(known_class_names) - set(pseudo_class_names))
        if count_mean_expected_steps:
            for class_name in all_class_names:
                classification_metrics[class_name]['mean_expected_steps'] = get_mean_expected_steps(
                    n_true_labels=n_true_labels,
                    n_pred_labels_top_n=n_pred_labels_top_n,
                    n_pred_scores_top_n=n_pred_scores_top_n,
                    label=class_name,
                    step_penalty=step_penalty
                )
        _add_metrics_to_dict(
            classification_metrics=classification_metrics,
            labels=known_class_names,
            tops_n=tops_n,
            prefix_caption='known_',
            count_mean_expected_steps=count_mean_expected_steps
        )
        _add_metrics_to_dict(
            classification_metrics=classification_metrics,
            labels=known_class_names_without_pseudo_classes,
            tops_n=tops_n,
            prefix_caption='known_',
            postfix_caption='_without_pseudo_classes',
            count_mean_expected_steps=count_mean_expected_steps
        )
    else:
        count_mean_expected_steps = False
    _add_metrics_to_dict(
        classification_metrics=classification_metrics,
        labels=all_class_names,
        tops_n=tops_n,
        prefix_caption='all_',
        count_mean_expected_steps=count_mean_expected_steps
    )
    _add_metrics_to_dict(
        classification_metrics=classification_metrics,
        labels=class_names_without_pseudo_classes,
        tops_n=tops_n,
        prefix_caption='all_',
        postfix_caption='_without_pseudo_classes',
        count_mean_expected_steps=count_mean_expected_steps
    )

    df_classification_metrics = pd.DataFrame(classification_metrics, dtype=object).T
    df_classification_metrics.sort_values(by='support', ascending=False, inplace=True)
    df_classification_metrics_MES_column = (
        ['mean_expected_steps'] if 'mean_expected_steps' in df_classification_metrics.columns else []
    )
    df_classification_metrics_columns = ['support'] + df_classification_metrics_MES_column + [
        'precision', 'recall', 'f1_score', 'value'
    ] + [
        item for sublist in [[f'precision@{top_n}', f'recall@{top_n}'] for top_n in tops_n if top_n > 1]
        for item in sublist
    ] + ['TP', 'FP', 'FN']
    df_classification_metrics = df_classification_metrics[df_classification_metrics_columns]

    if known_class_names is not None:
        df_classification_metrics.loc[all_class_names, 'known'] = (
            df_classification_metrics.loc[all_class_names].index.isin(known_class_names)
        )
        df_classification_metrics.loc[all_class_names, 'pseudo'] = (
            df_classification_metrics.loc[all_class_names].index.isin(pseudo_class_names)
        )

    return df_classification_metrics
