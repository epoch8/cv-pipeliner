import _thread
import weakref
import sys
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import tensorflow as tf

from two_stage_pipeliner.inference_models.detection.core import DetectionModelSpec
from two_stage_pipeliner.inference_models.detection.object_detection_api import (
    ObjectDetectionAPI_ModelSpec,
    ObjectDetectionAPI_pb_ModelSpec,
    ObjectDetectionAPI_TFLite_ModelSpec
)
from two_stage_pipeliner.inference_models.classification.core import ClassificationModelSpec
from two_stage_pipeliner.inference_models.classification.tensorflow import TensorFlow_ClassificationModelSpec
from two_stage_pipeliner.inference_models.pipeline import PipelineModelSpec
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer

from yacs.config import CfgNode
from .config import (
    object_detection_api,
    object_detection_api_pb,
    object_detection_api_tflite,
    tensorflow_cls_model
)

import streamlit as st


HASH_FUNCS = {
    tf.python.util.object_identity._ObjectIdentityWrapper: id,
    tf.python.util.object_identity._WeakObjectIdentityWrapper: id,
    type(weakref.ref(type({}))): id,
    weakref.KeyedRef: id,
    type({}.keys()): id,
    _thread.RLock: id,
    _thread.LockType: id,
    _thread._local: id,
    type(tf.compat.v1.get_variable_scope()): id,
    tf.python.framework.ops.EagerTensor: id
}


@dataclass
class DetectionModelDefinition:
    description: str
    model_spec: DetectionModelSpec
    score_threshold: float


@dataclass
class ClassificationDefinition:
    description: str
    model_spec: ClassificationModelSpec


def get_list_cfg_from_dict(d: Dict):
    return [item for sublist in [(str(k), str(v)) for k, v in d.items()] for item in sublist]


def get_cfg_from_dict(d: Dict, possible_cfgs: List[CfgNode]):
    assert len(d) == 1
    key = list(d)[0]
    cfg = None
    for possible_cfg in possible_cfgs:
        if set(dict(d[key])) == set(dict(possible_cfg)):
            cfg = possible_cfg.clone()
            cfg.merge_from_list(get_list_cfg_from_dict(d[key]))

    if cfg is None:
        raise ValueError(f'Got unknown config: {d}')
    return cfg, key


@st.cache(show_spinner=False)
def get_description_to_detection_model_definition_from_config(
    cfg: CfgNode
) -> Dict[str, DetectionModelDefinition]:
    detection_models_definitions = []
    for i, detection_cfg in enumerate(cfg.models.detection):
        detection_cfg, key = get_cfg_from_dict(
            d=detection_cfg,
            possible_cfgs=[object_detection_api, object_detection_api_pb, object_detection_api_tflite]
        )
        detection_model_definition = DetectionModelDefinition(
            description=f"[{i}] {detection_cfg.description}",
            score_threshold=detection_cfg.score_threshold,
            model_spec=None
        )
        if key == 'object_detection_api':
            object_detection_api
            detection_model_definition.model_spec = ObjectDetectionAPI_ModelSpec(
                config_path=detection_cfg.config_path,
                checkpoint_path=detection_cfg.checkpoint_path
            )
        elif key == 'object_detection_api_pb':
            detection_model_definition.model_spec = ObjectDetectionAPI_pb_ModelSpec(
                saved_model_dir=detection_cfg.saved_model_dir,
                input_type=detection_cfg.input_type
            )
        elif key == 'object_detection_api_tflite':
            detection_model_definition.model_spec = ObjectDetectionAPI_TFLite_ModelSpec(
                model_path=detection_cfg.model_path,
                bboxes_output_index=detection_cfg.bboxes_output_index,
                scores_output_index=detection_cfg.scores_output_index
            )
        detection_models_definitions.append(detection_model_definition)

    description_to_detection_model_definition = {
        detection_model_definition.description: detection_model_definition
        for detection_model_definition in detection_models_definitions
    }

    return description_to_detection_model_definition


def get_preprocess_input_from_script_file(script_file):
    parent_dir_of_script = Path(script_file).parent.absolute()
    sys.path.append(str(parent_dir_of_script))
    module_name = parent_dir_of_script.name
    module = importlib.import_module(module_name)
    sys.path.pop()
    return module.preprocess_input


@st.cache(show_spinner=False)
def get_description_to_classification_model_definition_from_config(
    cfg: CfgNode
) -> Dict[str, ClassificationDefinition]:
    classification_models_definitions = []
    for i, classification_cfg in enumerate(cfg.models.classification):
        classification_cfg, key = get_cfg_from_dict(
            d=classification_cfg,
            possible_cfgs=[tensorflow_cls_model]
        )
        classification_model_definition = ClassificationDefinition(
            description=f"[{i}] {classification_cfg.description}",
            model_spec=None
        )
        if key == 'tensorflow_cls_model':
            classification_model_definition.model_spec = TensorFlow_ClassificationModelSpec(
                input_size=classification_cfg.input_size,
                preprocess_input=get_preprocess_input_from_script_file(classification_cfg.preprocess_input_script_file),
                class_names=classification_cfg.class_names,
                model_path=classification_cfg.model_path,
                saved_model_type=classification_cfg.saved_model_type
            )
        classification_models_definitions.append(classification_model_definition)

    description_to_classification_model_definition = {
        classification_model_definition.description: classification_model_definition
        for classification_model_definition in classification_models_definitions
    }

    return description_to_classification_model_definition


@st.cache(hash_funcs=HASH_FUNCS, allow_output_mutation=True, show_spinner=False)
def load_pipeline_inferencer(
    detection_model_spec: DetectionModelSpec,
    classification_model_spec: ClassificationModelSpec
) -> PipelineInferencer:

    pipeline_model_spec = PipelineModelSpec(
        detection_model_spec=detection_model_spec,
        classification_model_spec=classification_model_spec
    )
    pipeline_model = pipeline_model_spec.load()
    pipeline_inferencer = PipelineInferencer(pipeline_model)

    return pipeline_inferencer
