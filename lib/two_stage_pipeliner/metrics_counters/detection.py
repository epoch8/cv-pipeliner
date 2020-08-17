from typing import Tuple

import pandas as pd

from two_stage_pipeliner.metrics_counters.core.detection import get_df_detector_metrics, \
    get_df_detector_metrics_recall
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.core.metrics_counter import MetricsCounter
from two_stage_pipeliner.inferencers.detection import DetectionInferencer


class DetectionMetricsCounter(MetricsCounter):
    def __init__(self, inferencer: DetectionInferencer):
        assert isinstance(inferencer, DetectionInferencer)
        MetricsCounter.__init__(self, inferencer)

    def score(self,
              data_generator: BatchGeneratorImageData,
              score_threshold: float,
              minimum_iou: float,
              return_recall_metrics: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            score_threshold=score_threshold
        )
        raw_n_pred_images_data = self.inferencer.predict(
            data_generator,
            score_threshold=0.
        )
        n_true_bboxes_data = [
            image_data.bboxes_data
            for image_data in n_true_images_data
        ]
        n_pred_bboxes_data = [
            image_data.bboxes_data
            for image_data in n_pred_images_data
        ]
        raw_n_pred_bboxes_data = [
            image_data.bboxes_data
            for image_data in raw_n_pred_images_data
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
        raw_n_pred_bboxes = [
            [
                (bbox_data.ymin, bbox_data.xmin, bbox_data.ymax, bbox_data.xmax)
                for bbox_data in bboxes_data
            ]
            for bboxes_data in raw_n_pred_bboxes_data
        ]
        raw_n_pred_scores = [
            [
                bbox_data.detection_score
                for bbox_data in bboxes_data
            ]
            for bboxes_data in raw_n_pred_bboxes_data
        ]
        df_detector_metrics = get_df_detector_metrics(
            n_true_bboxes,
            n_pred_bboxes,
            minimum_iou,
            raw_n_pred_bboxes,
            raw_n_pred_scores
        )
        n_true_labels = [
            [
                bbox_data.label
                for bbox_data in bboxes_data
            ]
            for bboxes_data in n_true_bboxes_data
        ]
        if return_recall_metrics:
            df_detector_metrics_recall = get_df_detector_metrics_recall(
                n_true_bboxes,
                n_pred_bboxes,
                n_true_labels,
                minimum_iou
            )
            return df_detector_metrics, df_detector_metrics_recall
        
        return df_detector_metrics
