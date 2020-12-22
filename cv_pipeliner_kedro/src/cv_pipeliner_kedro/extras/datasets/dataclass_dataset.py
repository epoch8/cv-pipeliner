import json
import importlib
from typing import Any, Dict, List, Type
from dataclasses import dataclass, asdict
from dacite import from_dict

import fsspec

from kedro.io import AbstractDataSet

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.utils.streamlit.data import get_images_data_from_dir
from cv_pipeliner.data_converters.brickit import BrickitDataConverter
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec, ModelSpec
from cv_pipeliner.inference_models.detection.object_detection_api import (
    ObjectDetectionAPI_ModelSpec
)
from cv_pipeliner.inference_models.classification.tensorflow import TensorFlow_ClassificationModelSpec


@dataclass
class PipelineModelSpec_Fix(ModelSpec):
    detection_model_spec: ObjectDetectionAPI_ModelSpec
    classification_model_spec: TensorFlow_ClassificationModelSpec

    @property
    def inference_model_cls(self) -> Type['PipelineModel']:
        from cv_pipeliner.inference_models.pipeline import PipelineModel
        return PipelineModel


class DataClassDataSet(AbstractDataSet):
    def __init__(self, data_class: str, filepath: str):
        split = data_class.split('.')
        module, function = '.'.join(split[:-1]), split[-1]
        module = importlib.import_module(module)
        self.data_class = eval(f"module.{function}")
        self.filepath = filepath
        if self.data_class is PipelineModelSpec:
            self.data_class = PipelineModelSpec_Fix

    def _load(self) -> Any:
        with fsspec.open(self.filepath, 'r') as src:
            data = json.load(src)
        print(f'{self.data_class=}, {self.filepath=}')
        dataclass = from_dict(
            data_class=self.data_class,
            data=data,
        )
        return dataclass

    def _save(self, data_class: Any) -> None:
        data = asdict(data_class)
        with fsspec.open(self.filepath, 'w') as out:
            json.dump(data, out, indent=4)

    def _describe(self) -> Dict[str, Any]:
        dict(
            data_class=self.data_class,
            filepath=self.filepath
        )

    def _exists(self) -> bool:
        open_file = fsspec.open(self.filepath, 'r')
        return open_file.fs.exists(self.filepath)
