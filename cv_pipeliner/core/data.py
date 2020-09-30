import io
from dataclasses import dataclass
from typing import Union, List
from pathlib import Path

import imageio
import numpy as np
import cv2


@dataclass(frozen=True)
class BboxData:
    image_path: Union[str, Path] = None
    image_bytes: io.BytesIO = None
    image: np.ndarray = None
    cropped_image: np.ndarray = None
    xmin: int = None
    ymin: int = None
    xmax: int = None
    ymax: int = None
    detection_score: float = None
    label: str = None
    classification_score: float = None

    top_n: int = None
    labels_top_n: List[str] = None
    classification_scores_top_n: List[float] = None

    def __post_init__(self):
        if self.image_path is not None:
            super().__setattr__('image_path', Path(self.image_path))

    def open_cropped_image(
        self,
        inplace: bool = False,
        source_image: np.ndarray = None,
        width_offset: int = 0,
        height_offset: int = 0
    ) -> Union[None, np.ndarray]:

        if self.cropped_image is not None:
            if not inplace:
                return self.cropped_image
            else:
                cropped_image = self.cropped_image
        else:
            if source_image is not None:
                image = source_image
            else:
                image = self.open_image()

            assert self.xmin < self.xmax and self.ymin < self.ymax

            height, width, _ = image.shape
            width_offset = max(0, min(width_offset, self.xmin-width_offset, width-self.xmax))
            height_offset = max(0, min(height_offset, self.ymin-height_offset, height-self.ymax))

            cropped_image = image[self.ymin-height_offset:self.ymax+height_offset,
                                  self.xmin-width_offset:self.xmax+width_offset]

        if inplace:
            super().__setattr__('cropped_image', cropped_image)
        else:
            return cropped_image

    def open_image(
        self,
        inplace: bool = False
    ) -> Union[None, np.ndarray]:
        if self.image is not None:
            if not inplace:
                return self.image
            else:
                image = self.image.copy()
        elif self.image_path is not None:
            image = np.array(imageio.imread(self.image_path, pilmode="RGB"))
        elif self.image_bytes is not None:
            image = np.array(imageio.imread(self.image_bytes))
        else:
            raise ValueError("BboxData doesn't have any image.")

        if inplace:
            super().__setattr__('image', image)
        else:
            return image

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
        if self.bboxes_data is None:
            super().__setattr__('bboxes_data', [])

    def open_image(self, inplace: bool = False) -> Union[None, np.ndarray]:
        if self.image is not None:
            if not inplace:
                return self.image
            else:
                image = self.image.copy()
        elif self.image_path is not None:
            image = np.array(imageio.imread(self.image_path, pilmode="RGB"))
        elif self.image_bytes is not None:
            image = np.array(imageio.imread(self.image_bytes))
            if image.shape[-1] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            if len(image.shape) == 2 or image.shape[-1] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("ImageData doesn't have any image.")

        if inplace:
            super().__setattr__('image', image)
        else:
            return image
