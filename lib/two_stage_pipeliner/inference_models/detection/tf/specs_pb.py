from dataclasses import dataclass
from typing import Union, Literal, Tuple, ClassVar
from pathlib import Path

from two_stage_pipeliner.inference_models.detection.core import DetectionModelSpec


@dataclass
class DetectionModelSpecTF_pb(DetectionModelSpec):
    spec_name: str
    input_size: Tuple[int, int]
    saved_model_dir: Union[str, Path]
    input_type: Literal["image_tensor", "float_image_tensor"]

    def __post_init__(self):
        self.saved_model_dir = Path(self.saved_model_dir).absolute()

    @property
    def inference_model(self) -> ClassVar['DetectionModelTF_pb']:
        from two_stage_pipeliner.inference_models.detection.tf.detector_pb import DetectionModelTF_pb
        return DetectionModelTF_pb
