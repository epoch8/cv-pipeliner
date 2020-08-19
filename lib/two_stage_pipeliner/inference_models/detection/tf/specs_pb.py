from dataclasses import dataclass
from typing import Union, Literal
from pathlib import Path

from two_stage_pipeliner.core.inference_model import Checkpoint


@dataclass
class DetectorModelSpecTF_pb(Checkpoint):
    input_size: int = None
    checkpoint_path: Union[str, Path] = None
    input_type: Literal["image_tensor", "float_image_tensor"] = None

    def __post_init__(self):
        self.checkpoint_path = Path(self.checkpoint_path).absolute()
