from dataclasses import dataclass
from typing import Union
from pathlib import Path


@dataclass
class DetectorModelSpecTF_pb:
    input_size: int = None
    checkpoint_path: Union[str, Path] = None
    input_type: Union["image_tensor", "float_image_tensor"] = None
