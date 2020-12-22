from typing import Dict, List
from pathy import Pathy

from cv_pipeliner.core.data import ImageData

from cv_pipeliner.utils.models_definitions import DetectionModelDefinition, ClassificationDefinition
from cv_pipeliner.inference_models.detection.object_detection_api import (
    ObjectDetectionAPI_ModelSpec
)
from cv_pipeliner.inference_models.classification.tensorflow import TensorFlow_ClassificationModelSpec
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec
from cv_pipeliner.reporters.pipeline import PipelineReporter

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple

import json
import fsspec
import pandas as pd
import nbformat as nbf

from cv_pipeliner.core.reporter import Reporter
from cv_pipeliner.core.data import ImageData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.inference_models.detection.core import DetectionModelSpec
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec, PipelineModel
from cv_pipeliner.inferencers.detection import DetectionInferencer
from cv_pipeliner.inferencers.pipeline import PipelineInferencer
from cv_pipeliner.metrics.detection import get_df_detection_metrics, df_detection_metrics_columns
from cv_pipeliner.metrics.pipeline import get_df_pipeline_metrics, df_pipeline_metrics_columns
from cv_pipeliner.visualizers.pipeline import PipelineVisualizer
from cv_pipeliner.logging import logger
from cv_pipeliner.utils.dataframes import transpose_columns_and_write_diffs_to_df_with_tags
from cv_pipeliner.utils.images_datas import cut_images_data_by_bboxes


def get_detection_model_definition_from_config(
    detection_model_definition_yaml: Dict
) -> DetectionModelDefinition:
    object_detection_api = detection_model_definition_yaml['object_detection_api']
    detection_model_definition = DetectionModelDefinition(
        description=object_detection_api['description'],
        score_threshold=object_detection_api['score_threshold'],
        model_spec=ObjectDetectionAPI_ModelSpec(
            config_path=object_detection_api['config_path'],
            checkpoint_path=object_detection_api['checkpoint_path'],
        ),
        model_index=object_detection_api['model_index']
    )
    return detection_model_definition


def get_classification_model_definition_from_config(
    classification_model_definition_yaml: Dict
) -> ClassificationDefinition:
    tensorflow_cls_model = classification_model_definition_yaml['tensorflow_cls_model']
    classification_model_definition = ClassificationDefinition(
        description=tensorflow_cls_model['description'],
        model_spec=TensorFlow_ClassificationModelSpec(
            input_size=eval(tensorflow_cls_model['input_size']),
            preprocess_input=tensorflow_cls_model['preprocess_input_script_file'],
            class_names=tensorflow_cls_model['class_names'],
            model_path=tensorflow_cls_model['model_path'],
            saved_model_type=tensorflow_cls_model['saved_model_type']
        ),
        model_index=tensorflow_cls_model['model_index']
    )

    return classification_model_definition


def get_model_spec(model_definition):
    return model_definition.model_spec


def get_detection_score_threshold(model_definition):
    return model_definition.score_threshold


def get_model(model_spec):
    return model_spec.load()


def get_model_class_names(model_spec):
    if isinstance(model_spec.class_names, List):
        return model_spec.class_names
    elif isinstance(model_spec.class_names, str):
        with fsspec.open(model_spec.class_names, 'r') as src:
            model_class_names = json.load(src)
        return model_class_names
    else:
        raise ValueError


def get_pipeline_model(detection_model, classification_model):
    pipeline_model = PipelineModel()
    pipeline_model.load_from_loaded_models(detection_model, classification_model)
    return pipeline_model


def make_detection_inference(inferencer, images_data_gen, score_threshold):
    return inferencer.predict(images_data_gen, score_threshold=score_threshold)


def make_raw_detection_inference(inferencer, images_data_gen):
    return inferencer.predict(images_data_gen, score_threshold=0.)


def make_pipeline_inference(inferencer, images_data_gen, detection_score_threshold):
    return inferencer.predict(images_data_gen, detection_score_threshold=detection_score_threshold)


def report(
    true_images_data: List[ImageData],
    pred_images_data_detection: List[ImageData],
    pred_images_data_detection_raw: List[ImageData],
    pred_images_data_pipeline: List[ImageData],
    tag: List[str],
    extra_bbox_label: List[str],
    compare_tag: str,
    output_directory: Union[str, Path],
    minimum_iou: float,
    pseudo_class_names: List[str],
):
    tag_df_pipeline_metrics = get_df_pipeline_metrics(
        true_images_data=true_images_data,
        pred_images_data=pred_images_data_pipeline,
        minimum_iou=minimum_iou,
        extra_bbox_label=extra_bbox_label,
        pseudo_class_names=pseudo_class_names,
        known_class_names=pipeline_model.class_names
    )
    tag_df_pipeline_metrics = _inference_pipeline_and_get_metrics(
        model_spec=model_spec,
        true_images_data=true_images_data,
        detection_score_threshold=detection_score_threshold,
        minimum_iou=minimum_iou,
        extra_bbox_label=extra_bbox_label,
        pseudo_class_names=pseudo_class_names,
        batch_size=batch_size,
        cut_by_bboxes=cut_by_bboxes
    )

    pipeline_report_data = PipelineReportData(
        df_detection_metrics=tag_df_detection_metrics,
        df_pipeline_metrics=tag_df_pipeline_metrics,
        tag=tag
    )

    return pipeline_report_data


def count_metrics(
    images_data: List[ImageData],
    detection_model_definition: Dict,
    classification_model_definition: Dict,
    pseudo_class_names: List[str],
    extra_bbox_label: str,
    minimum_iou: float,
    evaluate_batch_size: int,
    pipeline_report_dir: str,
):
    detection_model_definition = _get_detection_model_definition_from_config(detection_model_definition)
    classification_model_definition = _get_classification_model_definition_from_config(classification_model_definition)
    pipeline_model_spec = PipelineModelSpec(
        detection_model_spec=detection_model_definition.model_spec,
        classification_model_spec=classification_model_definition.model_spec
    )
    tag = f"{detection_model_definition.model_index}+{classification_model_definition.model_index}"
    pipeline_reporter = PipelineReporter()
    pipeline_reporter.report(
        models_specs=[pipeline_model_spec],
        tags=[tag],
        detection_scores_thresholds=[detection_model_definition.score_threshold],
        extra_bbox_labels=[extra_bbox_label],
        compare_tag=tag,
        output_directory=str(Pathy(pipeline_report_dir) / tag),
        true_images_data=images_data,
        minimum_iou=minimum_iou,
        pseudo_class_names=pseudo_class_names,
        batch_size=evaluate_batch_size
    )
