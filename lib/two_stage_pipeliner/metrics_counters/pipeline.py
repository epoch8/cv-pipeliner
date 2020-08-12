from typing import List
import pandas as pd

from two_stage_pipeliner.metrics_counters.core.pipeline import get_df_pipeline_metrics
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.core.metrics_counter import MetricsCounter
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer


class PipelineMetricsCounter(MetricsCounter):
    def __init__(self, inferencer: PipelineInferencer):
        assert isinstance(inferencer, PipelineInferencer)
        MetricsCounter.__init__(self, inferencer)

    def score(self,
              data_generator: BatchGeneratorImageData,
              detection_score_threshold: float,
              minimum_iou: float,
              soft: bool = True,
              known_class_names: List[str] = None,
              extra_bbox_label: str = "") -> pd.DataFrame:

        n_true_images_data = data_generator.data

        for image_data in n_true_images_data:
            assert image_data.bboxes_data is not None
            for bbox_data in image_data.bboxes_data:
                assert bbox_data.xmin is not None
                assert bbox_data.ymin is not None
                assert bbox_data.xmax is not None
                assert bbox_data.ymax is not None
                assert bbox_data.label is not None

        n_pred_images_data = self.inferencer.predict(
            data_generator,
            detection_score_threshold=detection_score_threshold
        )
        n_true_bboxes_data = [
            image_data.bboxes_data
            for image_data in n_true_images_data
        ]
        n_pred_bboxes_data = [
            image_data.bboxes_data
            for image_data in n_pred_images_data
        ]
        n_true_bboxes = [
            [
                (bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax)
                for bbox_data in bboxes_data
            ]
            for bboxes_data in n_true_bboxes_data
        ]
        n_pred_bboxes = [
            [
                (bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax)
                for bbox_data in bboxes_data
            ]
            for bboxes_data in n_pred_bboxes_data
        ]
        n_true_labels = [
            [
                bbox_data.label
                for bbox_data in bboxes_data
            ]
            for bboxes_data in n_true_bboxes_data
        ]
        n_pred_labels = [
            [
                bbox_data.label
                for bbox_data in bboxes_data
            ]
            for bboxes_data in n_pred_bboxes_data
        ]
        df_pipeline_metrics = get_df_pipeline_metrics(
            n_true_bboxes,
            n_true_labels,
            n_pred_bboxes,
            n_pred_labels,
            minimum_iou,
            soft,
            known_class_names,
            extra_bbox_label
        )
        return df_pipeline_metrics
