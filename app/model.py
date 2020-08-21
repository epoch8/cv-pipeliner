import json
import _thread
import weakref
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import streamlit as st
import tensorflow as tf

from brickit_ml.default_tf_settings import default_tf_settings
from two_stage_pipeliner.inference_models.detection.core import DetectionModelSpec
from two_stage_pipeliner.inference_models.classification.core import ClassificationModelSpec
from two_stage_pipeliner.inference_models.detection.tf.specs import (
    load_detection_model_spec_tf_from_standard_list_of_models_specs
)
from two_stage_pipeliner.inference_models.classification.tf.specs import (
    load_classification_model_spec_tf_from_standard_list_of_models_specs
)
from two_stage_pipeliner.inference_models.pipeline import PipelineModelSpec
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer


@dataclass
class DetectionModelDescription:
    name: str
    model_spec: DetectionModelSpec
    score_threshold: float


@dataclass
class ClassificationDescription:
    name: str
    model_spec: ClassificationModelSpec


MAIN_PATH = Path(__file__).parent.parent.absolute()
MODELS_PATH = MAIN_PATH / 'models'
name_to_detection_description = {
    spec.name: spec for spec in [
        DetectionModelDescription(
            name='brickit-ml => 200716_centernet_R101_test',
            model_spec=load_detection_model_spec_tf_from_standard_list_of_models_specs(
                spec_name='centernet_resnet101_v1_fpn_512x512_coco17_tpu-8',
                config_path=MODELS_PATH/'brickit-ml/detection/200716_centernet_R101_test/pipeline.config',
                checkpoint_path=MODELS_PATH/'brickit-ml/detection/200716_centernet_R101_test/checkpoint/ckpt-8',
            ),
            score_threshold=0.4
        ),
        DetectionModelDescription(
            name='crpt-ml => 2020-07-subclasses-recycling',
            model_spec=load_detection_model_spec_tf_from_standard_list_of_models_specs(
                spec_name='ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
                config_path=MODELS_PATH/'crpt-ml/detection/2020-07-subclasses-recycling/pipeline.config',
                checkpoint_path=MODELS_PATH/'crpt-ml/detection/2020-07-subclasses-recycling/checkpoint/ckpt-50'
            ),
            score_threshold=0.3
        )
    ]
}
name_to_classification_description = {
    spec.name: spec for spec in [
        ClassificationDescription(
            name='brickit-ml => 200427_2236__balanced_sampling_max300_dataloader_v2',
            model_spec=load_classification_model_spec_tf_from_standard_list_of_models_specs(
                spec_name='ResNet50_(224x224)',
                class_names=json.load(open(MODELS_PATH/'brickit-ml/classification/200427_2236__balanced_sampling_max300_dataloader_v2/class_names.json')),
                model_path=MODELS_PATH/'brickit-ml/classification/200427_2236__balanced_sampling_max300_dataloader_v2/best_model.h5'
            ),
        ),
        ClassificationDescription(
            name='crpt-ml => 2020-07-subclasses-recycling',
            model_spec=load_classification_model_spec_tf_from_standard_list_of_models_specs(
                spec_name='EfficientNetB0_no_padding',
                class_names=json.load(open(MODELS_PATH/'crpt-ml/classification/2020-07-subclasses-recycling/class_names.json')),
                model_path=MODELS_PATH/'crpt-ml/classification/2020-07-subclasses-recycling/best_model.hdf5'
            )
        )
    ]
}

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


@st.cache(hash_funcs=HASH_FUNCS, allow_output_mutation=True)
def load_pipeline_inferencer(detection_model_name, classification_model_name) -> PipelineInferencer:
    default_tf_settings()
    detection_model_spec = name_to_detection_description[detection_model_name].model_spec
    classification_model_spec = name_to_classification_description[classification_model_name].model_spec
    pipeline_model_spec = PipelineModelSpec(
        detection_model_spec=detection_model_spec,
        classification_model_spec=classification_model_spec
    )
    pipeline_model = pipeline_model_spec.load()
    pipeline_inferencer = PipelineInferencer(pipeline_model)
    return pipeline_inferencer
