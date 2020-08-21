import copy
from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Dict, Callable, Tuple, ClassVar
from functools import partial

import cv2
import numpy as np
import tensorflow as tf

from albumentations import LongestMaxSize, PadIfNeeded, Normalize
from efficientnet.tfkeras import EfficientNetB0, preprocess_input as efn_preprocess_input

from two_stage_pipeliner.inference_models.classification.core import ClassificationModelSpec


@dataclass
class ClassificationModelSpecTF(ClassificationModelSpec):
    name: str
    input_size: Tuple[int, int]
    preprocess_input: Callable[[List[np.ndarray], Tuple[int, int]], np.ndarray]
    num_classes: int = None
    class_names: List[str] = None
    model_path: Union[str, Path, tf.keras.Model] = None
    load_default_model: Callable[[int], tf.keras.Model] = None

    @property
    def inference_model(self) -> ClassVar['ClassificationModelTF']:
        from two_stage_pipeliner.inference_models.classification.tf.classifier import ClassificationModelTF
        return ClassificationModelTF

# ResNet50


def load_model_resnet50(num_classes: int, input_size: Tuple[int, int]) -> tf.keras.Model:
    width, height = input_size
    base_model = tf.keras.applications.resnet50.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(width, height, 3)
    )

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropout_layer = tf.keras.layers.Dropout(rate=0.5)
    prediction_layer = tf.keras.layers.Dense(num_classes,
                                             activation='softmax')

    model = tf.keras.Sequential([
      base_model,
      global_average_layer,
      dropout_layer,
      prediction_layer
    ])

    return model


def preprocess_input_resnet50(
    input: List[np.ndarray],
    input_size: Tuple[int, int]
):
    width, height = input_size
    max_size = max(width, height)
    resize = LongestMaxSize(max_size=max_size,
                            interpolation=cv2.INTER_LANCZOS4)
    padding = PadIfNeeded(height, width,
                          border_mode=cv2.BORDER_REPLICATE,
                          value=0)
    preprocess_input = Normalize()
    input = np.array(
        [resize(image=np.array(item))['image']
         if max(np.array(item).shape[0], np.array(item).shape[1]) >= max_size else np.array(item)
         for item in input]
    )
    input = np.array(
        [padding(image=np.array(item))['image'] for item in input]
    )
    preprocess_input = Normalize()
    input = preprocess_input(image=input)['image']
    return input


# EfficientNetB0


def load_EfficientNetB0_model(num_classes: int) -> tf.keras.Model:
    model = EfficientNetB0(weights='imagenet')

    x = model.layers[-3].output
    x = tf.keras.layers.Dense(num_classes,
                              activation='softmax',
                              kernel_regularizer='l2',
                              bias_regularizer='l2',
                              name='softmax')(x)

    model = tf.keras.models.Model(model.input, x)
    return model


def preprocess_input_efn(
    input: List[np.ndarray],
    input_size: Tuple[int, int]
):
    input = [
        cv2.resize(np.array(item), dsize=input_size) for item in input
    ]
    input = np.array(input)
    input = efn_preprocess_input(input)
    return input


# ModelSpec


spec_name_to_classification_model_spec_tf: Dict[str, ClassificationModelSpecTF] = {
    spec.name: spec for spec in [
        ClassificationModelSpecTF(
            name='ResNet50_(224x224)',
            load_default_model=partial(load_model_resnet50, input_size=(224, 224)),
            input_size=(224, 224),
            preprocess_input=partial(preprocess_input_resnet50, input_size=(224, 224)),
        ),
        ClassificationModelSpecTF(
            name='EfficientNetB0_no_padding',
            load_default_model=load_EfficientNetB0_model,
            input_size=(224, 224),
            preprocess_input=partial(preprocess_input_efn, input_size=(224, 224)),
        ),
    ]
}


def load_classification_model_spec_tf_from_standard_list_of_models_specs(
    spec_name: Union[str, Path],
    class_names: List[str],
    model_path: Union[str, Path] = None
) -> ClassificationModelSpecTF:

    model_spec = copy.deepcopy(spec_name_to_classification_model_spec_tf[spec_name])
    model_spec.class_names = class_names
    model_spec.num_classes = len(class_names)
    model_spec.model_path = Path(model_path).absolute() if model_path else None

    return model_spec
