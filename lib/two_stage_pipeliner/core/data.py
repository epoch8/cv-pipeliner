from dataclasses import dataclass
from typing import Union, List
from pathlib import Path

import numpy as np


@dataclass
class BboxData:
    image_path: Path = None
    image_bbox: np.ndarray = None
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    label: str = None
    detector_score: float = None
    classifier_score: float = None


@dataclass
class ImageData:
    image_path: Union[str, Path] = None
    image: np.ndarray = None
    bboxes_data: List[BboxData] = None
