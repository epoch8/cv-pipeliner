from dataclasses import dataclass
from typing import Literal, List

import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.logging import logger


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


def pairwise_intersection_over_union(
    bboxes_data1: List[BboxData],  # N
    bboxes_data2: List[BboxData]  # K
) -> List[List[float]]:  # Matrix NxK
    low = np.s_[..., :2]
    high = np.s_[..., 2:]
    A = np.array([
        [bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax]
        for bbox_data in bboxes_data1
    ])
    B = np.array([
        [bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax]
        for bbox_data in bboxes_data2
    ])
    A = A[:, None]
    B = B[None]
    A[high] += 1
    B[high] += 1
    interactions = np.prod(
        np.maximum(0, np.minimum(A[high], B[high]) - np.maximum(A[low], B[low])),
        axis=-1
    )
    bboxes_areas1 = np.prod(A[high] - A[low], axis=-1)
    bboxes_areas2 = np.prod(B[high] - B[low], axis=-1)
    return interactions / (bboxes_areas1 + bboxes_areas2 - interactions + 1e-9)


@dataclass
class BboxDataMatching:
    '''
    A dataclass providing macthing between true_bbox_data and pred_bbox_data inside of given image_data.
    By using this matching, we can get error type (TP, TN, FP, FN) for Detection or Pipeline models.
    '''
    true_bbox_data: BboxData = None
    pred_bbox_data: BboxData = None
    extra_bbox_label: str = None

    def __post_init__(self):
        if self.extra_bbox_label is None:
            self.extra_bbox_label = "trash"
        self._iou = intersection_over_union(self.true_bbox_data, self.pred_bbox_data)

    def get_detection_error_type(
        self,
        label: str = None
    ) -> Literal["TP", "TN", "FP", "FN"]:
        if label is not None and self.true_bbox_data is not None and \
                self.true_bbox_data.label != label:
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
        label: str = None
    ) -> Literal["TP", "TN", "FP", "FN", "TP (extra bbox)", "TN (extra bbox)", "FP (extra bbox)", "FN (extra bbox)"]:

        for bbox_data in [self.true_bbox_data, self.pred_bbox_data]:
            if bbox_data is not None:
                bbox_data.assert_label_is_valid()

        true_label = self.true_bbox_data.label if self.true_bbox_data is not None else None
        pred_label = self.pred_bbox_data.label if self.pred_bbox_data is not None else None

        assert true_label is not None or pred_label is not None

        # true_bbox is found:
        if self.true_bbox_data is not None and self.pred_bbox_data is not None:
            if label is None:
                if true_label != pred_label:
                    return "FP"
                else:
                    return "TP"
            else:
                if true_label != label and pred_label != label:
                    return "TN"
                elif true_label == label and pred_label == label:
                    return "TP"
                elif true_label != label and pred_label == label:
                    return "FP"
                else:
                    return "FN"
        # true_bbox is not found:
        elif self.true_bbox_data is not None and self.pred_bbox_data is None:
            if label is None:
                return "FN"
            else:
                if true_label == label:
                    return "FN"
                else:
                    return "TN"
        # pred_bbox is an extra
        elif self.true_bbox_data is None and self.pred_bbox_data is not None:
            if label is None:
                if pred_label != self.extra_bbox_label:
                    return "FP (extra bbox)"
                else:
                    return "TP (extra bbox)"
            else:
                if label != self.extra_bbox_label and pred_label != label:
                    return "TN (extra bbox)"
                elif label == self.extra_bbox_label and pred_label == label:
                    return "TP (extra bbox)"
                elif label != self.extra_bbox_label and pred_label == label:
                    return "FP (extra bbox)"
                else:
                    return "FN (extra bbox)"

    @property
    def iou(self):
        return self._iou


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
        extra_bbox_label: str = None,
        bboxes_data_matchings: List[BboxDataMatching] = None
    ):
        self.true_image_data = true_image_data
        self.pred_image_data = pred_image_data
        self.minimum_iou = minimum_iou
        self.extra_bbox_label = extra_bbox_label
        if bboxes_data_matchings is None:
            self.bboxes_data_matchings = self._get_bboxes_data_matchings(
                true_image_data=true_image_data,
                pred_image_data=pred_image_data,
                minimum_iou=minimum_iou,
                extra_bbox_label=extra_bbox_label
            )
        else:
            self.bboxes_data_matchings = bboxes_data_matchings

    def _get_bboxes_data_matchings(
        self,
        true_image_data: ImageData,
        pred_image_data: ImageData,
        minimum_iou: float,
        extra_bbox_label: str
    ) -> List[BboxDataMatching]:

        true_bboxes_data = true_image_data.bboxes_data
        pred_bboxes_data = pred_image_data.bboxes_data
        remained_pred_bboxes_data = set(range(len(pred_bboxes_data)))
        bboxes_data_matchings = []

        for tag, bboxes_data in [('true', true_bboxes_data),
                                 ('pred', pred_bboxes_data)]:
            bboxes_coords = set()
            for bbox_data in bboxes_data:
                bbox_data.assert_coords_are_valid()
                xmin, ymin, xmax, ymax = bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax
                if (xmin, ymin, xmax, ymax) in bboxes_coords:
                    logger.warning(
                        f'Repeated {tag} BboxData with these coords '
                        f'(xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}. '
                        'All BboxData should contain unique elements.'
                    )
                bboxes_coords.add((xmin, ymin, xmax, ymax))

        if len(true_bboxes_data) > 0 and len(bboxes_data) > 0:
            pairwise_ious = pairwise_intersection_over_union(true_bboxes_data, bboxes_data)

            for idx, true_bbox_data in enumerate(true_bboxes_data):
                best_pred_bbox_column = np.argmax(pairwise_ious[idx, :])

                if pairwise_ious[idx, best_pred_bbox_column] >= minimum_iou:
                    # Remove pred_bbox_data from pairwise matrix
                    pairwise_ious[:, best_pred_bbox_column] = -1
                    remained_pred_bboxes_data.remove(best_pred_bbox_column)
                else:
                    best_pred_bbox_column = None

                if best_pred_bbox_column is not None:  # Not found
                    bboxes_data_matchings.append(BboxDataMatching(
                        true_bbox_data=true_bbox_data,
                        pred_bbox_data=pred_bboxes_data[best_pred_bbox_column],
                        extra_bbox_label=extra_bbox_label
                    ))
                else:
                    bboxes_data_matchings.append(BboxDataMatching(  # Found
                        true_bbox_data=true_bbox_data,
                        pred_bbox_data=None,
                        extra_bbox_label=extra_bbox_label
                    ))
            for pred_bbox_data_column in remained_pred_bboxes_data:
                bboxes_data_matchings.append(BboxDataMatching(
                    true_bbox_data=None,
                    pred_bbox_data=pred_bboxes_data[pred_bbox_data_column],
                    extra_bbox_label=extra_bbox_label
                ))
        elif len(true_bboxes_data) > 0:
            for true_bbox_data in true_bboxes_data:
                bboxes_data_matchings.append(BboxDataMatching(
                    true_bbox_data=true_bbox_data,
                    pred_bbox_data=None,
                    extra_bbox_label=extra_bbox_label
                ))
        elif len(bboxes_data) > 0:
            for pred_bbox_data in bboxes_data:
                bboxes_data_matchings.append(BboxDataMatching(
                    true_bbox_data=None,
                    pred_bbox_data=pred_bbox_data,
                    extra_bbox_label=extra_bbox_label
                ))

        return bboxes_data_matchings

    def get_detection_errors_types(
        self,
        label: str = None
    ) -> List[Literal["TP", "FP", "FN"]]:
        detection_errors_types = [
            bbox_data_matching.get_detection_error_type(
                label=label
            )
            for bbox_data_matching in self.bboxes_data_matchings
        ]
        detection_errors_types = [error_type for error_type in detection_errors_types if error_type is not None]
        return detection_errors_types

    def get_pipeline_errors_types(
        self,
        label: str = None
    ) -> List[Literal["TP", "FP", "FN", "TP (extra bbox)", "FP (extra bbox)"]]:
        pipeline_errors_types = [
            bbox_data_matching.get_pipeline_error_type(
                label=label
            )
            for bbox_data_matching in self.bboxes_data_matchings
        ]
        pipeline_errors_types = [error_type for error_type in pipeline_errors_types if error_type is not None]
        return pipeline_errors_types

    def get_detection_TP(self, label: str = None) -> int:
        return self.get_detection_errors_types(label).count("TP")

    def get_detection_FP(self) -> int:
        return self.get_detection_errors_types().count("FP")

    def get_detection_FN(self, label: str = None) -> int:
        return self.get_detection_errors_types(label).count("FN")

    def get_pipeline_TP(
        self,
        label: str = None
    ) -> int:
        return self.get_pipeline_errors_types(label).count("TP")

    def get_pipeline_FP(
        self,
        label: str = None
    ) -> int:
        return self.get_pipeline_errors_types(label).count("FP")

    def get_pipeline_FN(
        self,
        label: str = None
    ) -> int:
        return self.get_pipeline_errors_types(label).count("FN")

    def get_pipeline_TP_extra_bbox(
        self,
        label: str = None
    ) -> int:
        return self.get_pipeline_errors_types(label).count("TP (extra bbox)")

    def get_pipeline_FP_extra_bbox(
        self,
        label: str = None
    ) -> int:
        return self.get_pipeline_errors_types(label).count("FP (extra bbox)")

    def get_pipeline_FN_extra_bbox(
        self,
        label: str = None
    ) -> int:
        return self.get_pipeline_errors_types(label).count("FN (extra bbox)")

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
