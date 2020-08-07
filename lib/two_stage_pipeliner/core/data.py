from dataclasses import dataclass
from typing import Union, List
from pathlib import Path

import numpy as np


@dataclass
class BboxData:
    image_path: Path = None
    image_bbox: np.ndarray = None
    xmin: int = None
    ymin: int = None
    xmax: int = None
    ymax: int = None
    detection_score: float = None
    label: str = None
    classification_score: float = None


@dataclass
class ImageData:
    image_path: Union[str, Path] = None
    image: np.ndarray = None
    bboxes_data: List[BboxData] = None
