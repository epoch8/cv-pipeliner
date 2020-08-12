from dataclasses import dataclass
from typing import Union, List
from pathlib import Path

import imageio
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

    def open_image_bbox(self, inplace: bool = False) -> Union[None, np.ndarray]:
        image = imageio.imread(self.image_path, pilmode="RGB")

        assert self.xmin < self.xmax and self.ymin < self.ymax

        image_bbox = image[self.ymin:self.ymax,
                           self.xmin:self.xmax]
        if inplace:
            self.image_bbox = image_bbox
        else:
            return image_bbox

@dataclass
class ImageData:
    image_path: Union[str, Path] = None
    image: np.ndarray = None
    bboxes_data: List[BboxData] = None

    def open_image(self, inplace: bool = False) -> Union[None, np.ndarray]:
        image = imageio.imread(self.image_path, pilmode="RGB")
        if inplace:
            self.image = image
        else:
            return image
