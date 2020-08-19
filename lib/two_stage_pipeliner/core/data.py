import io
from dataclasses import dataclass
from typing import Union, List
from pathlib import Path

import imageio
import numpy as np


@dataclass(frozen=True)
class BboxData:
    image_path: Union[str, Path] = None
    image_bytes: io.BytesIO = None
    cropped_image: np.ndarray = None
    xmin: int = None
    ymin: int = None
    xmax: int = None
    ymax: int = None
    detection_score: float = None
    label: str = None
    classification_score: float = None

    def __post_init__(self):
        if self.image_path is not None:
            super().__setattr__('image_path', Path(self.image_path))

    def open_cropped_image(self, inplace: bool = False) -> Union[None, np.ndarray]:
        if not inplace and self.cropped_image is not None:
            cropped_image = self.cropped_image.copy()
        else:
            if self.image_path is not None:
                image = np.array(imageio.imread(self.image_path, pilmode="RGB"))
            elif self.image_bytes is not None:
                image = np.array(imageio.imread(self.image_bytes))
            else:
                raise ValueError("BboxData doesn't have any image.")

            assert self.xmin < self.xmax and self.ymin < self.ymax

            cropped_image = image[self.ymin:self.ymax,
                                  self.xmin:self.xmax]

        if inplace:
            super().__setattr__('cropped_image', cropped_image)
        else:
            return cropped_image

    @property
    def visualize_label(self):
        return self.visualize_label

    def assert_coords_are_valid(self):
        assert all(x is not None for x in [self.xmin, self.ymin, self.xmax, self.ymax])
        assert self.xmin <= self.xmax and self.ymin <= self.ymax

    def assert_label_is_valid(self):
        assert self.label is not None


@dataclass(frozen=True)
class ImageData:
    image_path: Union[str, Path] = None
    image_bytes: io.BytesIO = None
    image: np.ndarray = None
    bboxes_data: List[BboxData] = None

    def __post_init__(self):
        if self.image_path is not None:
            super().__setattr__('image_path', Path(self.image_path))

    def open_image(self, inplace: bool = False) -> Union[None, np.ndarray]:
        if self.image is not None:
            image = self.image.copy()
        elif self.image_path is not None:
            image = np.array(imageio.imread(self.image_path, pilmode="RGB"))
        elif self.image_bytes is not None:
            image = np.array(imageio.imread(self.image_bytes))
        else:
            raise ValueError("ImageData doesn't have any image.")

        if inplace:
            super().__setattr__('image', image)
        else:
            return image
