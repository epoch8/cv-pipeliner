import copy
import logging
import os
import sys

from pathlib import Path
from typing import Literal, Optional, Union, Dict, Tuple

from cv_pipeliner.core.data import ImageData, BboxData
from cv_pipeliner.metrics.image_data_matching import ImageDataMatching
from cv_pipeliner.utils.images_datas import flatten_additional_bboxes_data_in_image_data

logger = logging.getLogger('cv_pipeliner.utils.fiftyone')


class FifyOneSession:
    _counter = 0

    def __init__(self, fiftyone_database_dir: Union[str, Path] = None):
        assert FifyOneSession._counter == 0, (
            "There is another instance of class FifyOneConverter. Delete it for using this class."
        )
        if 'fiftyone' in sys.modules:
            logger.warning(
                'Fiftyone is already imported to some base. The FifyOneConverter will reimport fiftyone and it can change the base.'
            )
            del sys.modules['fiftyone']
        self.fiftyone_database_dir = fiftyone_database_dir
        if self.fiftyone_database_dir is not None:
            os.environ['FIFTYONE_DATABASE_DIR'] = str(self.fiftyone_database_dir)
        import fiftyone
        self.fiftyone = fiftyone

        FifyOneSession._counter += 1

    def __del__(self):
        if self.fiftyone_database_dir is not None:
            del os.environ['FIFTYONE_DATABASE_DIR']
        del self.fiftyone
        del sys.modules['fiftyone']
        FifyOneSession._counter -= 1

    def convert_bbox_data_to_fo_detection(self, bbox_data: BboxData) -> 'fiftyone.Detection':
        xminn, yminn, xmaxn, ymaxn = bbox_data.coords_n
        bounding_box = [xminn, yminn, xmaxn-xminn, ymaxn-yminn]
        return self.fiftyone.Detection(label=bbox_data.label, bounding_box=bounding_box)

    def convert_image_data_to_fo_detections(
        self, image_data: ImageData,
        include_additional_bboxes_data: bool = False
    ) -> 'fiftyone.Detections':
        if include_additional_bboxes_data:
            image_data = flatten_additional_bboxes_data_in_image_data(image_data)

        return self.fiftyone.Detections(
            detections=list(map(self.convert_bbox_data_to_fo_detection, image_data.bboxes_data))
        )

    def convert_image_data_to_fo_detections_by_classes(
        self,
        image_data: ImageData,
        include_additional_bboxes_data: bool = False
    ) -> Dict[str, 'fiftyone.Detections']:
        if include_additional_bboxes_data:
            image_data = flatten_additional_bboxes_data_in_image_data(image_data)
        class_names = sorted(set([bbox_data.label for bbox_data in image_data.bboxes_data]))
        class_name_to_fo_detections = {
            class_name: self.fiftyone.Detections(
                detections=list(map(self.convert_bbox_data_to_fo_detection, [
                    bbox_data for bbox_data in image_data.bboxes_data
                    if bbox_data.label == class_name
                ]))
            )
            for class_name in class_names
        }
        return class_name_to_fo_detections

    def convert_image_data_matching_to_fo_detections(
        self,
        image_data_matching: ImageDataMatching,
        model_type: Literal['detection', 'pipeline'],
        label: Optional[str] = None
    ) -> Tuple['fiftyone.Detections', 'fiftyone.Detections']:
        """
            Convert image_data_matching to FO.Detections, with labels written as "<label> [<error type>]".
            Returns pair of true and pred detections: (true_detections, pred_detections)
        """
        assert model_type in ['detection', 'pipeline']
        image_data_matching = copy.deepcopy(image_data_matching)

        true_detections_instances = []
        pred_detections_instances = []
        for bbox_data_matching in image_data_matching.bboxes_data_matchings:
            if model_type == 'detection':
                bbox_error_type = bbox_data_matching.get_detection_error_type(label=label)
            else:
                bbox_error_type = bbox_data_matching.get_pipeline_error_type(label=label)
            true_bbox_data = bbox_data_matching.true_bbox_data
            pred_bbox_data = bbox_data_matching.pred_bbox_data
            if true_bbox_data is not None:
                true_bbox_data.label = f"{true_bbox_data.label} [{bbox_error_type}]"
                true_detections_instances.append(self.convert_bbox_data_to_fo_detection(true_bbox_data))
            if pred_bbox_data is not None:
                pred_bbox_data.label = f"{pred_bbox_data.label} [{bbox_error_type}]"
                pred_detections_instances.append(self.convert_bbox_data_to_fo_detection(pred_bbox_data))

        true_detections = self.fiftyone.Detections(detections=true_detections_instances)
        pred_detections = self.fiftyone.Detections(detections=pred_detections_instances)
        return true_detections, pred_detections

    def convert_image_data_matching_to_fo_detections_by_classes(
        self,
        image_data_matching: ImageDataMatching,
        model_type: Literal['detection', 'pipeline'],
        label: Optional[str] = None
    ) -> Tuple[Dict[str, 'fiftyone.Detections'], Dict[str, 'fiftyone.Detections']]:
        """
            Convert image_data_matching to FO.Detections, where bboxes_data_matchings are matched by classes.

            If argument 'label' is set, errors types counts as
            binary classification of [0, 1], where 1 is 'label' and 0 is the other class. 

            Returns part of dict [true_dict, pred_dict] of this format:
            {
                'class_type [error_type]': fiftyone.Detections
            }
        """
        assert model_type in ['detection', 'pipeline']
        errors_types = ['TP', 'FP', 'FN', 'TN', 'TP (extra bbox)', 'FP (extra bbox)']
        class_names = sorted(set([
            bbox_data_matching.true_bbox_data.label
            for bbox_data_matching in image_data_matching.bboxes_data_matchings
            if bbox_data_matching.true_bbox_data is not None
        ] + [
            bbox_data_matching.pred_bbox_data.label
            for bbox_data_matching in image_data_matching.bboxes_data_matchings
            if bbox_data_matching.pred_bbox_data is not None
        ]))
        class_name_to_fo_true_detections = {
            f"{class_name} [{error_type}]": [] for class_name in class_names for error_type in errors_types
        }
        class_name_to_fo_pred_detections = {
            f"{class_name} [{error_type}]": [] for class_name in class_names for error_type in errors_types
        }
        for bbox_data_matching in image_data_matching.bboxes_data_matchings:
            if model_type == 'detection':
                bbox_error_type = bbox_data_matching.get_detection_error_type(label=label)
            else:
                bbox_error_type = bbox_data_matching.get_pipeline_error_type(label=label)

            true_bbox_data = bbox_data_matching.true_bbox_data
            pred_bbox_data = bbox_data_matching.pred_bbox_data
            if true_bbox_data is not None:
                class_name_to_fo_true_detections[f"{true_bbox_data.label} [{bbox_error_type}]"].append(
                    self.convert_bbox_data_to_fo_detection(true_bbox_data)
                )
            if pred_bbox_data is not None:
                class_name_to_fo_pred_detections[f"{pred_bbox_data.label} [{bbox_error_type}]"].append(
                    self.convert_bbox_data_to_fo_detection(pred_bbox_data)
                )

        for key in class_name_to_fo_true_detections:
            class_name_to_fo_true_detections[key] = self.fiftyone.Detections(
                detections=class_name_to_fo_true_detections[key]
            )
        for key in class_name_to_fo_pred_detections:
            class_name_to_fo_pred_detections[key] = self.fiftyone.Detections(
                detections=class_name_to_fo_pred_detections[key]
            )

        return class_name_to_fo_true_detections, class_name_to_fo_pred_detections
