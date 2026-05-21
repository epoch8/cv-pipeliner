import copy
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.metrics.image_data_matching import ImageDataMatching
from cv_pipeliner.utils.images_datas import flatten_additional_bboxes_data_in_image_data

# from threading import Semaphore


logger = logging.getLogger("cv_pipeliner.utils.fiftyone")


class FiftyOneSession:
    _active_sessions = 0
    _env_var_by_param = {
        "database_dir": "FIFTYONE_DATABASE_DIR",
        "database_uri": "FIFTYONE_DATABASE_URI",
        "database_name": "FIFTYONE_DATABASE_NAME",
    }

    def __init__(
        self,
        database_dir: Optional[Union[str, Path]] = None,
        database_uri: Optional[str] = None,
        database_name: Optional[str] = None,
    ):
        if FiftyOneSession._active_sessions > 0:
            raise RuntimeError(
                "There is another active FiftyOneSession. Close it before creating a new session."
            )
        self.database_dir = database_dir
        self.database_uri = database_uri
        self.database_name = database_name
        self._previous_env_values = {}
        self._closed = False
        self._fiftyone = None

        if self._has_database_config() and "fiftyone" in sys.modules:
            logger.warning(
                "FiftyOne is already imported, so database settings may have already been initialized. "
                "Create FiftyOneSession before importing fiftyone to guarantee custom database settings."
            )

        self._set_environment()

        try:
            self._fiftyone = importlib.import_module("fiftyone")
        except Exception as e:
            logger.warning(f"Couldn't import fiftyone: {e=}")

        FiftyOneSession._active_sessions += 1

    def _has_database_config(self):
        return any(getattr(self, param_name) is not None for param_name in self._env_var_by_param)

    def _set_environment(self):
        for param_name, env_var_name in self._env_var_by_param.items():
            value = getattr(self, param_name)
            if value is None:
                continue
            self._previous_env_values[env_var_name] = os.environ.get(env_var_name)
            os.environ[env_var_name] = str(value)

    def _restore_environment(self):
        for env_var_name, previous_value in self._previous_env_values.items():
            if previous_value is None:
                os.environ.pop(env_var_name, None)
            else:
                os.environ[env_var_name] = previous_value

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._restore_environment()
        self._closed = True
        FiftyOneSession._active_sessions -= 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def fiftyone(self):
        if self._fiftyone is None:
            self._fiftyone = importlib.import_module("fiftyone")
        return self._fiftyone

    def __del__(self):
        self.close()

    def convert_bbox_data_to_fo_detection(
        self,
        bbox_data: BboxData,
        additional_info_keys: Optional[List[str]] = None,
    ) -> "fiftyone.Detection":
        additional_info_keys = additional_info_keys or []
        xminn, yminn, xmaxn, ymaxn = bbox_data.coords_n
        bounding_box = [xminn, yminn, xmaxn - xminn, ymaxn - yminn]
        additional_info = {key: bbox_data.additional_info.get(key, None) for key in additional_info_keys}
        return self.fiftyone.Detection(label=bbox_data.label, bounding_box=bounding_box, **additional_info)

    def convert_bbox_data_keypoints_to_fo_keypoint(self, bbox_data: BboxData) -> "fiftyone.Keypoint":
        if len(bbox_data.keypoints) > 0:
            return self.fiftyone.Keypoint(
                label=bbox_data.label,
                points=[tuple(pair) for pair in bbox_data.keypoints_n],
                source_coords=bbox_data.coords,  # FIXME: https://github.com/voxel51/fiftyone/issues/1610
            )
        else:
            return None

    def convert_image_data_to_fo_detections(
        self,
        image_data: ImageData,
        include_additional_bboxes_data: bool = False,
        additional_info_keys: Optional[List[str]] = None,
    ) -> "fiftyone.Detections":
        additional_info_keys = additional_info_keys or []
        if include_additional_bboxes_data:
            image_data = flatten_additional_bboxes_data_in_image_data(image_data)
        image_data.get_image_size()  # Save to meta
        return self.fiftyone.Detections(
            detections=[
                self.convert_bbox_data_to_fo_detection(bbox_data, additional_info_keys)
                for bbox_data in image_data.bboxes_data
            ]
        )

    def convert_image_data_to_fo_keypoints(
        self, image_data: ImageData, include_additional_bboxes_data: bool = False
    ) -> "fiftyone.Detections":
        if include_additional_bboxes_data:
            image_data = flatten_additional_bboxes_data_in_image_data(image_data)
        image_data.get_image_size()  # Save to meta
        keypoints = (
            [self.fiftyone.Keypoint(label=image_data.label, points=[tuple(pair) for pair in image_data.keypoints_n])]
            if len(image_data.keypoints) > 0
            else []
        ) + [
            fo_keypoints
            for fo_keypoints in map(self.convert_bbox_data_keypoints_to_fo_keypoint, image_data.bboxes_data)
            if fo_keypoints is not None
        ]

        return self.fiftyone.Keypoints(keypoints=keypoints)

    def convert_image_data_to_fo_detections_by_classes(
        self,
        image_data: ImageData,
        include_additional_bboxes_data: bool = False,
    ) -> Dict[str, "fiftyone.Detections"]:
        if include_additional_bboxes_data:
            image_data = flatten_additional_bboxes_data_in_image_data(image_data)
        class_names = sorted(set([bbox_data.label for bbox_data in image_data.bboxes_data]))
        class_name_to_fo_detections = {
            class_name: self.fiftyone.Detections(
                detections=list(
                    map(
                        self.convert_bbox_data_to_fo_detection,
                        [bbox_data for bbox_data in image_data.bboxes_data if bbox_data.label == class_name],
                    )
                )
            )
            for class_name in class_names
        }
        return class_name_to_fo_detections

    def convert_image_data_matching_to_fo_detections(
        self,
        image_data_matching: ImageDataMatching,
        model_type: Literal["detection", "pipeline"],
        label: Optional[str] = None,
    ) -> Tuple["fiftyone.Detections", "fiftyone.Detections"]:
        """
        Convert image_data_matching to FO.Detections, with labels written as "<label> [<error type>]".
        Returns pair of true and pred detections: (true_detections, pred_detections)
        """
        assert model_type in ["detection", "pipeline"]
        image_data_matching = copy.deepcopy(image_data_matching)

        true_detections_instances = []
        pred_detections_instances = []
        for bbox_data_matching in image_data_matching.bboxes_data_matchings:
            if model_type == "detection":
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
        model_type: Literal["detection", "pipeline"],
        label: Optional[str] = None,
    ) -> Tuple[Dict[str, "fiftyone.Detections"], Dict[str, "fiftyone.Detections"]]:
        """
        Convert image_data_matching to FO.Detections, where bboxes_data_matchings are matched by classes.

        If argument 'label' is set, errors types counts as
        binary classification of [0, 1], where 1 is 'label' and 0 is the other class.

        Returns part of dict [true_dict, pred_dict] of this format:
        {
            'class_type [error_type]': fiftyone.Detections
        }
        """
        assert model_type in ["detection", "pipeline"]
        errors_types = ["TP", "FP", "FN", "TN", "TP (extra bbox)", "FP (extra bbox)"]
        class_names = sorted(
            set(
                [
                    bbox_data_matching.true_bbox_data.label
                    for bbox_data_matching in image_data_matching.bboxes_data_matchings
                    if bbox_data_matching.true_bbox_data is not None
                ]
                + [
                    bbox_data_matching.pred_bbox_data.label
                    for bbox_data_matching in image_data_matching.bboxes_data_matchings
                    if bbox_data_matching.pred_bbox_data is not None
                ]
            )
        )
        class_name_to_fo_true_detections = {
            f"{class_name} [{error_type}]": [] for class_name in class_names for error_type in errors_types
        }
        class_name_to_fo_pred_detections = {
            f"{class_name} [{error_type}]": [] for class_name in class_names for error_type in errors_types
        }
        for bbox_data_matching in image_data_matching.bboxes_data_matchings:
            if model_type == "detection":
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

    def convert_fo_detection_to_bbox_data(
        self,
        fo_detection: "fiftyone.Detection",
        width: int,
        height: int,
        additional_info_keys_in_fo_detections: Optional[List[str]] = None,
        bbox_data_cls: Type[BboxData] = BboxData,
    ) -> BboxData:
        additional_info_keys_in_fo_detections = additional_info_keys_in_fo_detections or []
        xminn, yminn, widthn, heightn = fo_detection.bounding_box
        xmaxn, ymaxn = xminn + widthn, yminn + heightn
        xmin, xmax = xminn * width, xmaxn * width
        ymin, ymax = yminn * height, ymaxn * height
        return bbox_data_cls(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            meta_width=width,
            meta_height=height,
            label=fo_detection.label,
            additional_info={key: fo_detection[key] for key in additional_info_keys_in_fo_detections},
        )

    def convert_fo_keypoint_to_numpy_keypoints(
        self, fo_keypoint: "fiftyone.Keypoint", width: int, height: int
    ) -> np.array:
        keypoints = np.array(fo_keypoint.points).astype(float).reshape((-1, 2))
        keypoints[:, 0] *= width
        keypoints[:, 1] *= height
        return keypoints

    def convert_image_data_to_fo_sample(
        self,
        image_data: ImageData,
        fo_detections_label: Optional[str] = None,  # берется bboxes_data с их label
        fo_classification_label: Optional[str] = None,  # берется image_data.label
        fo_keypoints_label: Optional[str] = None,  # берется image_data.kepoints + bbox_data.keypoints
        include_additional_bboxes_data: bool = False,
        mapping_filepath: Callable[[str], str] = lambda filepath: filepath,
        additional_info_keys_in_bboxes_data: Optional[List[str]] = None,
        additional_info_keys_in_image_data: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> "fiftyone.Detections":
        additional_info_keys_in_bboxes_data = additional_info_keys_in_bboxes_data or []
        additional_info_keys_in_image_data = additional_info_keys_in_image_data or []
        additional_info = additional_info or {}
        filepath = mapping_filepath(str(image_data.image_path))
        sample = self.fiftyone.Sample(filepath=filepath)
        if fo_detections_label is not None:
            width, height = image_data.get_image_size()  # Save to meta
            sample[fo_detections_label] = self.convert_image_data_to_fo_detections(
                image_data, include_additional_bboxes_data, additional_info_keys_in_bboxes_data
            )
            sample.metadata = self.fiftyone.ImageMetadata(width=width, height=height)
        if fo_classification_label is not None and image_data.label is not None:
            sample[fo_classification_label] = self.fiftyone.Classification(label=image_data.label)
        if fo_keypoints_label is not None:
            sample[fo_keypoints_label] = self.convert_image_data_to_fo_keypoints(
                image_data, include_additional_bboxes_data
            )
        for key in additional_info_keys_in_image_data:
            sample[key] = image_data.additional_info.get(key, None)
        for key, value in additional_info.items():
            sample[key] = value
        return sample

    def convert_sample_to_image_data(
        self,
        sample: "fiftyone.Sample",
        fo_detections_label: Optional[str] = None,
        fo_classification_label: Optional[str] = None,
        fo_keypoints_label: Optional[str] = None,
        mapping_filepath: Callable[[str], str] = lambda filepath: filepath,
        additional_info_keys_in_fo_detections: Optional[List[str]] = None,
        additional_info_keys_in_sample: Optional[List[str]] = None,
        image_data_cls: Type[ImageData] = ImageData,
        bbox_data_cls: Type[BboxData] = BboxData,
    ) -> ImageData:
        additional_info_keys_in_fo_detections = additional_info_keys_in_fo_detections or []
        additional_info_keys_in_sample = additional_info_keys_in_sample or []
        image_path = mapping_filepath(sample.filepath)
        image_data = image_data_cls(
            image_path=image_path,
            meta_width=sample.metadata.width if sample.metadata else None,
            meta_height=sample.metadata.height if sample.metadata else None,
        )
        if fo_detections_label is not None and (
            sample.has_field(fo_detections_label) and sample[fo_detections_label] is not None
        ):
            width, height = image_data.get_image_size()
            image_data.bboxes_data = [
                self.convert_fo_detection_to_bbox_data(
                    fo_detection, width, height, additional_info_keys_in_fo_detections, bbox_data_cls=bbox_data_cls
                )
                for fo_detection in sample[fo_detections_label].detections
            ]
        if fo_keypoints_label is not None and (
            sample.has_field(fo_keypoints_label) and sample[fo_keypoints_label] is not None
        ):
            coords_to_idx = {bbox_data.coords: idx for idx, bbox_data in enumerate(image_data.bboxes_data)}
            for fo_keypoint in sample[fo_keypoints_label].keypoints:
                keypoints = self.convert_fo_keypoint_to_numpy_keypoints(fo_keypoint, width, height)
                if "source_coords" in fo_keypoint:  # FIXME: https://github.com/voxel51/fiftyone/issues/1610
                    image_data.bboxes_data[coords_to_idx[tuple(fo_keypoint.source_coords)]].keypoints = keypoints
                else:
                    image_data.keypoints = np.append(image_data.keypoints, keypoints)
        if fo_classification_label is not None and (
            sample.has_field(fo_classification_label) and sample[fo_classification_label] is not None
        ):
            image_data.label = sample[fo_classification_label].label
        for key in additional_info_keys_in_sample:
            if sample.has_field(key):
                image_data.additional_info[key] = sample[key]
        return image_data


FifyOneSession = FiftyOneSession
