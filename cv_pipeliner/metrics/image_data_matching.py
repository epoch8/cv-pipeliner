from dataclasses import dataclass
from typing import Literal, List

import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData


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
    By using this matching, we can get error type (TP, TN, FP, FN) for Detection or Pipeline models.
    '''
    true_bbox_data: BboxData = None
    pred_bbox_data: BboxData = None
    extra_bbox_label: str = None

    def get_detection_error_type(
        self,
        filter_by_label: str = None
    ) -> Literal["TP", "TN", "FP", "FN"]:
        if filter_by_label is not None and self.true_bbox_data is not None and \
                self.true_bbox_data.label != filter_by_label:
            return "TN"
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
        filter_by_label: str
    ) -> Literal["TP", "TN", "FP", "FN", "TP (extra bbox)", "TN (extra bbox)", "FP (extra bbox)", "FN (extra bbox)"]:

        for bbox_data in [self.true_bbox_data, self.pred_bbox_data]:
            if bbox_data is not None:
                bbox_data.assert_label_is_valid()

        true_label = self.true_bbox_data.label if self.true_bbox_data is not None else None
        pred_label = self.pred_bbox_data.label if self.pred_bbox_data is not None else None

        assert true_label is not None or pred_label is not None

        # true_bbox is found:
        if self.true_bbox_data is not None and self.pred_bbox_data is not None:
            if true_label != filter_by_label and pred_label != filter_by_label:
                return "TN"
            elif true_label == filter_by_label and pred_label == filter_by_label:
                return "TP"
            elif true_label != filter_by_label and pred_label == filter_by_label:
                return "FP"
            else:
                return "FN"
        # true_bbox is not found:
        elif self.true_bbox_data is not None and self.pred_bbox_data is None:
            if true_label == filter_by_label:
                return "FN"
            else:
                return "TN"
        # pred_bbox is an extra
        elif self.true_bbox_data is None and self.pred_bbox_data is not None:
            if filter_by_label != self.extra_bbox_label and pred_label != filter_by_label:
                return "TN (extra bbox)"
            elif filter_by_label == self.extra_bbox_label and pred_label == filter_by_label:
                return "TP (extra bbox)"
            elif filter_by_label != self.extra_bbox_label and pred_label == filter_by_label:
                return "FP (extra bbox)"
            else:
                return "FN (extra bbox)"

    @property
    def iou(self):
        return intersection_over_union(self.true_bbox_data, self.pred_bbox_data)


@dataclass(init=False)
class ImageDataMatching:
    '''
    A dataclass providing macthing between true_bboxes_data and pred_bboxes_data
    inside of given image_data.

    We say that pred_bbox_data is matched to true_bbox_data if they have iou >= minimum_iou.
    One true_bbox_data may have only one matching to the pred_bbox_data.

    For illustrations, look tests/test_image_data_matching.py
    '''
    true_image_data: ImageData
    pred_image_data: ImageData
    minimum_iou: float
    extra_bbox_label: str
    bboxes_data_matchings: List[BboxDataMatching]

    def __init__(
        self,
        true_image_data: ImageData,
        pred_image_data: ImageData,
        minimum_iou: float,
        extra_bbox_label: str = None
    ):
        self.true_image_data = true_image_data
        self.pred_image_data = pred_image_data
        self.minimum_iou = minimum_iou
        self.bboxes_data_matchings = self._get_bboxes_data_matchings(
            true_image_data=true_image_data,
            pred_image_data=pred_image_data,
            minimum_iou=minimum_iou,
            extra_bbox_label=extra_bbox_label
        )

    def _get_bboxes_data_matchings(
        self,
        true_image_data: ImageData,
        pred_image_data: ImageData,
        minimum_iou: float,
        extra_bbox_label: str = None
    ) -> List[BboxDataMatching]:

        true_bboxes_data = true_image_data.bboxes_data
        pred_bboxes_data = pred_image_data.bboxes_data
        remained_pred_bboxes_data = pred_bboxes_data.copy()
        bboxes_data_matchings = []

        for tag, bboxes_data in [('true', true_bboxes_data),
                                 ('pred', pred_bboxes_data)]:
            bboxes_coords = set()
            for bbox_data in bboxes_data:
                bbox_data.assert_coords_are_valid()
                xmin, ymin, xmax, ymax = bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax
                if (xmin, ymin, xmax, ymax) in bboxes_coords:
                    raise ValueError(
                        f'Repeated {tag} BboxData with these coords '
                        '(xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}. '
                        'All BboxData must contain unique elements.'
                    )
                bboxes_coords.add((xmin, ymin, xmax, ymax))

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
                    extra_bbox_label=extra_bbox_label
                ))
            else:
                bboxes_data_matchings.append(BboxDataMatching(
                    true_bbox_data=true_bbox_data,
                    pred_bbox_data=None,
                    extra_bbox_label=extra_bbox_label
                ))
        for pred_bbox_data in remained_pred_bboxes_data:
            bboxes_data_matchings.append(BboxDataMatching(
                true_bbox_data=None,
                pred_bbox_data=pred_bbox_data,
                extra_bbox_label=extra_bbox_label
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
        filter_by_label: List[str]
    ) -> List[Literal["TP", "FP", "FN", "TP (extra bbox)", "FP (extra bbox)"]]:
        pipeline_errors_types = [
            bbox_data_matching.get_pipeline_error_type(
                filter_by_label=filter_by_label
            )
            for bbox_data_matching in self.bboxes_data_matchings
        ]
        pipeline_errors_types = [error_type for error_type in pipeline_errors_types if error_type is not None]
        return pipeline_errors_types

    def get_detection_TP(self, filter_by_label: str) -> int:
        return self.get_detection_errors_types(filter_by_label).count("TP")

    def get_detection_FP(self) -> int:
        return self.get_detection_errors_types().count("FP")

    def get_detection_FN(self, filter_by_label: str) -> int:
        return self.get_detection_errors_types(filter_by_label).count("FN")

    def get_pipeline_TP(
        self,
        filter_by_label: str
    ) -> int:
        return self.get_pipeline_errors_types(filter_by_label).count("TP")

    def get_pipeline_FP(
        self,
        filter_by_label: str
    ) -> int:
        return self.get_pipeline_errors_types(filter_by_label).count("FP")

    def get_pipeline_FN(
        self,
        filter_by_label: str
    ) -> int:
        return self.get_pipeline_errors_types(filter_by_label).count("FN")

    def get_pipeline_TP_extra_bbox(
        self,
        filter_by_label: str
    ) -> int:
        return self.get_pipeline_errors_types(filter_by_label).count("TP (extra bbox)")

    def get_pipeline_FP_extra_bbox(
        self,
        filter_by_label: str
    ) -> int:
        return self.get_pipeline_errors_types(filter_by_label).count("FP (extra bbox)")

    def find_bbox_data_matching(
        self,
        bbox_data: BboxData,
        tag: Literal['true', 'pred']
    ) -> BboxDataMatching:
        xmin, ymin, xmax, ymax = bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax
        if tag == 'true':
            bboxes_data_coords_from_matchings = [
                (
                    bbox_data_matching.true_bbox_data.xmin,
                    bbox_data_matching.true_bbox_data.ymin,
                    bbox_data_matching.true_bbox_data.xmax,
                    bbox_data_matching.true_bbox_data.ymax
                )
                if bbox_data_matching.true_bbox_data is not None
                else (-1, -1, -1, 1)
                for bbox_data_matching in self.bboxes_data_matchings
            ]
        elif tag == 'pred':
            bboxes_data_coords_from_matchings = [
                (
                    bbox_data_matching.pred_bbox_data.xmin,
                    bbox_data_matching.pred_bbox_data.ymin,
                    bbox_data_matching.pred_bbox_data.xmax,
                    bbox_data_matching.pred_bbox_data.ymax
                )
                if bbox_data_matching.pred_bbox_data is not None
                else (-1, -1, -1, 1)
                for bbox_data_matching in self.bboxes_data_matchings
            ]
        bbox_data_matching_index = bboxes_data_coords_from_matchings.index((xmin, ymin, xmax, ymax))

        return self.bboxes_data_matchings[bbox_data_matching_index]
