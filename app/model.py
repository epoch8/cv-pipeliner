import json
import _thread
import weakref
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import streamlit as st
import tensorflow as tf

from brickit_ml.default_tf_settings import default_tf_settings
from two_stage_pipeliner.inference_models.detection.tf.specs import load_detector_model_spec_tf
from two_stage_pipeliner.inference_models.detection.load_checkpoint import (
    load_detection_model_from_checkpoint
)
from two_stage_pipeliner.inference_models.classification.tf.specs import load_classifier_model_spec_tf
from two_stage_pipeliner.inference_models.classification.load_checkpoint import (
    load_classification_model_from_checkpoint
)
from two_stage_pipeliner.inference_models.pipeline import Pipeline
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer


@dataclass
class DetectionModelSpec:
    name: str
    checkpoint: Any
    score_threshold: float


@dataclass
class ClassificationModelSpec:
    name: str
    checkpoint: Any


MAIN_PATH = Path('.').absolute().parent
MODELS_PATH = MAIN_PATH / 'models'
name_to_detection_model_spec = {
    spec.name: spec for spec in [
        DetectionModelSpec(
            name='brickit-ml => 200716_centernet_R101_test',
            checkpoint=load_detector_model_spec_tf(
                model_name='centernet_resnet101_v1_fpn_512x512_coco17_tpu-8',
                model_dir=MODELS_PATH/'brickit-ml/detection/200716_centernet_R101_test/',
                checkpoint_filename='ckpt-8'
            ),
            score_threshold=0.4
        ),
        DetectionModelSpec(
            name='crpt-ml => 2020-07-subclasses-recycling',
            checkpoint=load_detector_model_spec_tf(
                model_name='ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8',
                model_dir=MODELS_PATH/'crpt-ml/detection/2020-07-subclasses-recycling/',
                checkpoint_filename='ckpt-50'
            ),
            score_threshold=0.3
        )
    ]
}
name_to_classification_model_spec = {
    spec.name: spec for spec in [
        ClassificationModelSpec(
            name='brickit-ml => 200427_2236__balanced_sampling_max300_dataloader_v2',
            checkpoint=load_classifier_model_spec_tf(
                model_name='ResNet50',
                class_names=json.load(open(MODELS_PATH/'brickit-ml/classification/200427_2236__balanced_sampling_max300_dataloader_v2/class_names.json')),
                model_path=MODELS_PATH/'brickit-ml/classification/200427_2236__balanced_sampling_max300_dataloader_v2/best_model.h5'
            ),
        ),
        ClassificationModelSpec(
            name='crpt-ml => 2020-07-subclasses-recycling',
            checkpoint=load_classifier_model_spec_tf(
                model_name='EfficientNetB0_no_padding',
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
    detection_model_spec = name_to_detection_model_spec[detection_model_name]
    classification_model_spec = name_to_classification_model_spec[classification_model_name]
    detection_model = load_detection_model_from_checkpoint(detection_model_spec.checkpoint)
    classification_model = load_classification_model_from_checkpoint(classification_model_spec.checkpoint)
    pipeline_model = Pipeline()
    pipeline_model.load((detection_model, classification_model))
    pipeline_inferencer = PipelineInferencer(pipeline_model)
    return pipeline_inferencer
