from dataclasses import dataclass
from typing import Literal, List

import numpy as np

from two_stage_pipeliner.core.data import BboxData, ImageData


def intersection_over_union(bbox_data1: BboxData, bbox_data2: BboxData) -> float:
    if bbox_data1 is None or bbox_data2 is None:
        return None
    xmins = np.max([bbox_data1.xmin, bbox_data2.xmin])
    ymins = np.max([bbox_data1.ymin, bbox_data2.ymin])
    xmaxs = np.min([bbox_data1.xmax, bbox_data2.xmax])
    ymaxs = np.min([bbox_data1.ymax, bbox_data2.ymax])
    inter_area = np.max([0, ymaxs - ymins + 1]) * np.max([0, xmaxs - xmins + 1])
    bbox1_area = (bbox_data1.xmax - bbox_data1.xmin + 1) * (bbox_data1.ymax - bbox_data1.ymin + 1)
    bbox2_area = (bbox_data2.xmax - bbox_data2.xmin + 1) * (bbox_data2.ymax - bbox_data2.ymin + 1)
    iou = inter_area / np.float(bbox1_area + bbox2_area - inter_area + 1e-9)
    return iou


@dataclass
class BboxDataMatching:
    '''
    A dataclass providing macthing between true_bbox_data and pred_bbox_data inside of given image_data.
    By using this matching, we can get error type (TP, FP, FN) for Detection or Pipeline models.

    For Detection:
        If true_bbox_data is matched to one pred_bbox_data,
            The matching is True Positive.

        If true_bbox_data is not matched to any pred_bbox_data (true_bbox_data isn't found),
            The matching is False Negative.

        If pred_bbox_data is not matched to any true_bbox_data (extra bbox),
            The matching is False Positive.

    For Pipeline:
        There are 2 ways to get error type: strict and soft.

        Classification model can know only part of all possible labels.
        We use List[str] of these labels and call it known_labels (argument 'use_soft_with_known_labels').
        Soft error type shows how pipeline works with using only it's known_labels.
        The only difference between Strict and Soft is an additional condition
            in "true_bbox_data is matched to the pred_bbox_data".


        (detection) If true_bbox_data is matched to the pred_bbox_data:
            Strict:
                (classification) If true_bbox_data.label == pred_bbox_data.label, then matching is True Positive.
                (classification) If true_bbox_data.label != pred_bbox_data.label, then matching is False Positive.

            Soft:
                If true_bbox_data.label in known_labels:
                    (classification) If true_bbox_data.label == pred_bbox_data.label, then matching is True Positive.
                    (classification) If true_bbox_data.label != pred_bbox_data.label, then matching is False Positive.
                If true_bbox_data.label not in known_labels and pred_bbox_data.label in known_labels:
                    Then matching is True Positive

        (detection) If true_bbox_data not matched to the pred_bbox_data (true_bbox_data isn't found):
            The matching is False Negative.

        (detection) If pred_bbox_data isn't matched to any true_bbox_data (extra bbox):
            If extra_bbox_label is given:
                (classification) If pred_bbox_data.label == extra_bbox_label, then matching is True Positive.
                (classification) If pred_bbox_data.label != extra_bbox_label, then matching is False Positive.
            If extra_bbox_label is not given:
                The matching is False Positive
    '''
    true_bbox_data: BboxData = None
    pred_bbox_data: BboxData = None

    def get_detection_error_type(
        self,
        filter_by_label: str = None
    ) -> Literal[None, "TP", "FP", "FN"]:
        if filter_by_label is not None and self.true_bbox_data is not None and \
                self.true_bbox_data.label != filter_by_label:
            return None
        # true_bbox is found:
        if self.true_bbox_data is not None and self.pred_bbox_data is not None:
            return "TP"
        # true_bbox is not found:
        elif self.true_bbox_data is not None and self.pred_bbox_data is None:
            return "FN"
        # pred_bbox is an extra:
        elif self.true_bbox_data is None and self.pred_bbox_data is not None:
            return "FP"

    def get_pipeline_error_type(
        self,
        filter_by_label: str = None,
        extra_bbox_label: str = None,
        use_soft_with_known_labels: List[str] = None
    ) -> Literal[None, "TP", "FP", "FN", "TP (extra bbox)", "FP (extra bbox)"]:

        true_label = self.true_bbox_data.label if self.true_bbox_data is not None else None
        pred_label = self.pred_bbox_data.label if self.pred_bbox_data is not None else None

        assert true_label is not None or pred_label is not None

        if filter_by_label is not None and \
                (true_label != filter_by_label or pred_label != filter_by_label):
            return None

        # true_bbox is found and labels are equal:
        if self.true_bbox_data is not None and self.pred_bbox_data is not None:
            if use_soft_with_known_labels is None:  # Strict
                if true_label == pred_label:
                    return "TP"
                else:
                    return "FP"
            else:  # Soft
                if true_label in use_soft_with_known_labels:
                    if true_label == pred_label:
                        return "TP"
                    else:
                        return "FP"
                else:
                    return "TP"

        # true_bbox is not found:
        elif self.true_bbox_data is not None and self.pred_bbox_data is None:
            return "FN"

        # pred_bbox is an extra
        elif self.true_bbox_data is None and self.pred_bbox_data is not None:
            # should be equal to extra_bbox_label if given, else it's FP
            if extra_bbox_label is not None and pred_label == extra_bbox_label:
                return "TP (extra bbox)"
            else:
                return "FP (extra bbox)"

    @property
    def iou(self):
        return intersection_over_union(self.true_bbox_data, self.pred_bbox_data)


@dataclass(init=False)
class ImageDataMatching:
    '''A dataclass providing macthing between true_bboxes_data and pred_bboxes_data
    inside of given image_data.

    We say that pred_bbox_data is matched to true_bbox_data if they have iou >= minimum_iou.
    One true_bbox_data may have only one matching to the pred_bbox_data).
    '''
    true_image_data: ImageData
    pred_image_data: ImageData
    minimum_iou: float
    bboxes_data_matchings: List[BboxDataMatching]

    def __init__(self, true_image_data: ImageData, pred_image_data: ImageData, minimum_iou: float):
        self.true_image_data = true_image_data
        self.pred_image_data = pred_image_data
        self.minimum_iou = minimum_iou
        self.bboxes_data_matchings = self._get_bboxes_data_matchings(
            true_image_data=true_image_data,
            pred_image_data=pred_image_data,
            minimum_iou=minimum_iou
        )

    def _get_bboxes_data_matchings(
        self,
        true_image_data: ImageData,
        pred_image_data: ImageData,
        minimum_iou: float
    ) -> List[BboxDataMatching]:

        true_bboxes_data = true_image_data.bboxes_data
        pred_bboxes_data = pred_image_data.bboxes_data
        remained_pred_bboxes_data = pred_bboxes_data.copy()
        bboxes_data_matchings = []

        def find_best_bbox_idx_by_iou(true_bbox_data: BboxData,
                                      pred_bboxes_data: List[BboxData],
                                      minimum_iou: float) -> int:
            if len(pred_bboxes_data) == 0:
                return None
            bboxes_iou = [
                intersection_over_union(true_bbox_data, pred_bbox_data)
                for pred_bbox_data in pred_bboxes_data
            ]
            best_pred_bbox_idx = np.argmax(bboxes_iou)

            best_iou = bboxes_iou[best_pred_bbox_idx]
            if best_iou >= minimum_iou:
                return best_pred_bbox_idx
            else:
                return None

        for true_bbox_data in true_bboxes_data:
            best_pred_bbox_idx = find_best_bbox_idx_by_iou(
                true_bbox_data, remained_pred_bboxes_data, minimum_iou
            )
            if best_pred_bbox_idx is not None:
                best_pred_bbox_data = remained_pred_bboxes_data.pop(best_pred_bbox_idx)
                bboxes_data_matchings.append(BboxDataMatching(
                    true_bbox_data=true_bbox_data,
                    pred_bbox_data=best_pred_bbox_data,
                ))
            else:
                bboxes_data_matchings.append(BboxDataMatching(
                    true_bbox_data=true_bbox_data,
                    pred_bbox_data=None,
                ))
        for pred_bbox_data in remained_pred_bboxes_data:
            bboxes_data_matchings.append(BboxDataMatching(
                true_bbox_data=None,
                pred_bbox_data=pred_bbox_data,
            ))
        return bboxes_data_matchings

    def get_detection_errors_types(
        self,
        filter_by_label=None
    ) -> List[Literal["TP", "FP", "FN"]]:
        detection_errors_types = [
            bbox_data_matching.get_detection_error_type(
                filter_by_label=filter_by_label
            )
            for bbox_data_matching in self.bboxes_data_matchings
        ]
        detection_errors_types = [error_type for error_type in detection_errors_types if error_type is not None]
        return detection_errors_types

    def get_pipeline_errors_types(
        self,
        filter_by_label: str = None,
        extra_bbox_label: str = None,
        use_soft_with_known_labels: List[str] = None
    ) -> List[Literal["TP", "FP", "FN",  "TP (extra bbox)", "FP (extra bbox)"]]:
        pipeline_errors_types = [
            bbox_data_matching.get_pipeline_error_type(
                filter_by_label=filter_by_label,
                extra_bbox_label=extra_bbox_label,
                use_soft_with_known_labels=use_soft_with_known_labels
            )
            for bbox_data_matching in self.bboxes_data_matchings
        ]
        pipeline_errors_types = [error_type for error_type in pipeline_errors_types if error_type is not None]
        return pipeline_errors_types

    def get_detection_TP(self, filter_by_label=None) -> int:
        return self.get_detection_errors_types(
            filter_by_label=filter_by_label
        ).count("TP")

    def get_detection_FP(self) -> int:
        return self.get_detection_errors_types().count("FP")

    def get_detection_FN(self, filter_by_label=None) -> int:
        return self.get_detection_errors_types(
            filter_by_label=filter_by_label
        ).count("FN")

    def get_pipeline_TP(
        self,
        filter_by_label: str = None,
        extra_bbox_label: str = None,
        use_soft_with_known_labels: List[str] = None
    ) -> int:
        return self.get_pipeline_errors_types(
            filter_by_label=filter_by_label,
            extra_bbox_label=extra_bbox_label,
            use_soft_with_known_labels=use_soft_with_known_labels
        ).count("TP")

    def get_pipeline_FP(
        self,
        filter_by_label: str = None,
        extra_bbox_label: str = None,
        use_soft_with_known_labels: List[str] = None
    ):
        return self.get_pipeline_errors_types(
            filter_by_label=filter_by_label,
            extra_bbox_label=extra_bbox_label,
            use_soft_with_known_labels=use_soft_with_known_labels
        ).count("FP")

    def get_pipeline_FN(
        self,
        filter_by_label: str = None,
        extra_bbox_label: str = None,
        use_soft_with_known_labels: List[str] = None
    ):
        return self.get_pipeline_errors_types(
            filter_by_label=filter_by_label,
            extra_bbox_label=extra_bbox_label,
            use_soft_with_known_labels=use_soft_with_known_labels
        ).count("FN")

    def get_pipeline_TP_extra_bbox(
        self,
        extra_bbox_label: str = None,
    ):
        return self.get_pipeline_errors_types(
            filter_by_label=None,
            extra_bbox_label=extra_bbox_label,
            use_soft_with_known_labels=None
        ).count("TP (extra bbox)")

    def get_pipeline_FP_extra_bbox(
        self,
        extra_bbox_label: str = None,
    ):
        return self.get_pipeline_errors_types(
            filter_by_label=None,
            extra_bbox_label=extra_bbox_label,
            use_soft_with_known_labels=None
        ).count("FP (extra bbox)")
