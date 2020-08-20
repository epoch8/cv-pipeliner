from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Dict, Callable, Tuple
from functools import partial

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from albumentations import LongestMaxSize, PadIfNeeded, Normalize
from efficientnet.tfkeras import EfficientNetB0, preprocess_input as efn_preprocess_input

from two_stage_pipeliner.core.inference_model import Checkpoint


@dataclass
class ClassifierModelSpecTF(Checkpoint):
    name: str
    input_size: Tuple[int, int]
    preprocess_input: Callable[[List[np.ndarray], Tuple[int, int]], np.ndarray]
    num_classes: int = None
    class_names: List[str] = None
    model_path: Path = None
    load_default_model: Callable[[int], tf.keras.Model] = None

# ResNet50


def load_model_resnet50(num_classes: int) -> tf.keras.Model:
    base_model = keras.applications.resnet50.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    global_average_layer = keras.layers.GlobalAveragePooling2D()
    dropout_layer = keras.layers.Dropout(rate=0.5)
    prediction_layer = keras.layers.Dense(len(num_classes),
                                          activation='softmax')

    model = keras.Sequential([
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
        cv2.resize(np.array(item), dsize=max_size) for item in input
    ]
    input = np.array(input)
    input = efn_preprocess_input(input)
    return input


# ModelSpec


name_to_model_spec: Dict[str, ClassifierModelSpecTF] = {
    spec.name: spec for spec in [
        ClassifierModelSpecTF(
            name='ResNet50',
            load_default_model=load_model_resnet50,
            input_size=(224, 224),
            preprocess_input=partial(preprocess_input_resnet50, input_size=(224, 224)),
        ),
        ClassifierModelSpecTF(
            name='EfficientNetB0_no_padding',
            load_default_model=load_EfficientNetB0_model,
            input_size=(224, 224),
            preprocess_input=partial(preprocess_input_efn, input_size=(224, 224)),
        ),
    ]
}


def load_classifier_model_spec_tf(
    model_name: Union[str, Path],
    class_names: List[str],
    model_path: Union[str, Path] = None
) -> ClassifierModelSpecTF:
    model_spec = name_to_model_spec[model_name]
    model_spec.class_names = class_names
    model_spec.num_classes = len(class_names)
    model_spec.model_path = Path(model_path).absolute() if model_path else None
    return model_spec
