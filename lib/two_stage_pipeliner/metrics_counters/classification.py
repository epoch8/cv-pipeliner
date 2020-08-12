import pandas as pd

from two_stage_pipeliner.metrics_counters.core.classification import get_df_classifier_metrics
from two_stage_pipeliner.core.batch_generator import BatchGeneratorBboxData
from two_stage_pipeliner.core.metrics_counter import MetricsCounter
from two_stage_pipeliner.inferencers.classification import ClassificationInferencer


class ClassificationMetricsCounter(MetricsCounter):
    def __init__(self, inferencer: ClassificationInferencer):
        assert isinstance(inferencer, ClassificationInferencer)
        MetricsCounter.__init__(self, inferencer)

    def score(self, data_generator: BatchGeneratorBboxData) -> pd.DataFrame:
        n_true_bboxes_data = data_generator.data
        n_pred_bboxes_data = self.inferencer.predict(data_generator)
        true_bboxes_data = [bbox_data for bboxes_data in n_true_bboxes_data for bbox_data in bboxes_data]
        pred_bboxes_data = [bbox_data for bboxes_data in n_pred_bboxes_data for bbox_data in bboxes_data]
        true_labels = [bbox_data.label for bbox_data in true_bboxes_data]
        pred_labels = [bbox_data.label for bbox_data in pred_bboxes_data]
        df_classifier_metrics = get_df_classifier_metrics(true_labels, pred_labels)
        return df_classifier_metrics
