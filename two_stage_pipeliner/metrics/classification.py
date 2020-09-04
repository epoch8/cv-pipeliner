from typing import List

import pandas as pd
from sklearn.metrics import classification_report

from two_stage_pipeliner.core.data import BboxData


def get_df_classification_metrics(
    n_true_bboxes_data: List[List[BboxData]],
    n_pred_bboxes_data: List[List[BboxData]]
) -> pd.DataFrame:
    assert len(n_true_bboxes_data) == len(n_true_bboxes_data)
    true_bboxes_data = [bbox_data for bboxes_data in n_true_bboxes_data for bbox_data in bboxes_data]
    pred_bboxes_data = [bbox_data for bboxes_data in n_pred_bboxes_data for bbox_data in bboxes_data]
    assert len(true_bboxes_data) == len(pred_bboxes_data)
    true_labels = [bbox_data.label for bbox_data in true_bboxes_data]
    pred_labels = [bbox_data.label for bbox_data in pred_bboxes_data]
    df_classifier_metrics = pd.DataFrame(
        classification_report(true_labels, pred_labels, output_dict=True),
        dtype=object
    ).T
    df_classifier_metrics.loc['accuracy', 'support'] = len(true_labels)
    df_classifier_metrics.sort_values(by='support', ascending=False, inplace=True)
    return df_classifier_metrics
