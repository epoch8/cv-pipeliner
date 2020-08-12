from typing import List

import pandas as pd
from sklearn.metrics import classification_report


def get_df_classifier_metrics(true_labels: List[str],
                              pred_labels: List[str]) -> pd.DataFrame:
    df_classifier_metrics = pd.DataFrame(
        classification_report(true_labels, pred_labels, output_dict=True)
    ).drop(columns='accuracy').T
    for val in ['precision', 'recall', 'f1-score']:
        df_classifier_metrics[val] = df_classifier_metrics[val].apply(
            lambda x: round(x, 3)
        )
    df_classifier_metrics['support'] = df_classifier_metrics['support'].astype(
        int
    )
    df_classifier_metrics.sort_values(
        by=f'support', ascending=False, inplace=True
    )

    return df_classifier_metrics
