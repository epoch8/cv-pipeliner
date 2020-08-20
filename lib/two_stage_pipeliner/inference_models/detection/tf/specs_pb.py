from dataclasses import dataclass
from typing import Union, Literal, Tuple
from pathlib import Path

from two_stage_pipeliner.core.inference_model import Checkpoint


@dataclass
class DetectorModelSpecTF_pb(Checkpoint):
    input_size: Tuple[int, int]
    checkpoint_path: Union[str, Path]
    input_type: Literal["image_tensor", "float_image_tensor"]

    def __post_init__(self):
        self.checkpoint_path = Path(self.checkpoint_path).absolute()
