from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Dict, Callable

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from albumentations import LongestMaxSize, PadIfNeeded, Normalize


@dataclass
class ClassifierModelSpecTF:
    name: str
    input_size: int
    preprocess_input: Callable[List[np.ndarray], np.ndarray]
    num_classes: int = None
    class_names: List[str] = None
    model_path: Path = None
    load_default_model: Callable[[int], tf.keras.Model] = None


def load_model_resnet50(num_classes: int):
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


def preprocess_input_resnet50(input: np.ndarray):
    resize = LongestMaxSize(max_size=224,
                            interpolation=cv2.INTER_LANCZOS4)
    padding = PadIfNeeded(224, 224,
                          border_mode=cv2.BORDER_REPLICATE,
                          value=0)
    preprocess_input = Normalize()
    input = np.array(
        [resize(image=np.array(item))['image'] for item in input]
    )
    input = np.array(
        [padding(image=np.array(item))['image'] for item in input]
    )
    preprocess_input = Normalize()
    input = preprocess_input(image=input)['image']
    return input


name_to_model_spec: Dict[str, ClassifierModelSpecTF] = {
    spec.name: spec for spec in [
        ClassifierModelSpecTF(
            name='ResNet50',
            load_default_model=load_model_resnet50,
            input_size=224,
            preprocess_input=preprocess_input_resnet50,
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
    model_spec.model_path = Path(model_path) if model_path else None
    return model_spec
