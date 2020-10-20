import io
from dataclasses import dataclass, field
from typing import Union, List, Dict, Tuple, Callable
from pathlib import Path

import imageio
import numpy as np
import cv2

from cv_pipeliner.utils.images import draw_rectangle


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

    additional_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.image_path is not None:
            super().__setattr__('image_path', Path(self.image_path))

    def open_cropped_image(
        self,
        inplace: bool = False,
        source_image: np.ndarray = None,
        xmin_offset: int = 0,
        ymin_offset: int = 0,
        xmax_offset: int = 0,
        ymax_offset: int = 0,
        draw_rectangle_with_color: Tuple[int, int, int] = None,
        thickness: int = 3,
        alpha: float = 0.3,
        return_as_bbox_data_in_cropped_image: bool = False,
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
            xmin_in_cropped_image = max(0, min(xmin_offset, self.xmin-xmin_offset))
            ymin_in_cropped_image = max(0, min(ymin_offset, self.ymin-ymin_offset))
            xmax_in_cropped_image = max(0, min(xmax_offset, width-self.xmax))
            ymax_in_cropped_image = max(0, min(ymax_offset, height-self.ymax))
            cropped_image = image[self.ymin-ymin_in_cropped_image:self.ymax+ymax_in_cropped_image,
                                  self.xmin-xmin_in_cropped_image:self.xmax+xmax_in_cropped_image]
            if draw_rectangle_with_color is not None:
                height, width, colors = cropped_image.shape
                cropped_image = draw_rectangle(
                    image=cropped_image,
                    xmin=xmin_in_cropped_image,
                    ymin=ymin_in_cropped_image,
                    xmax=width-xmax_in_cropped_image,
                    ymax=height-ymax_in_cropped_image,
                    color=draw_rectangle_with_color,
                    thickness=thickness,
                    alpha=alpha
                )
                # cv2.rectangle(
                #     img=cropped_image.copy(),
                #     pt1=(xmin_in_cropped_image, ymin_in_cropped_image),
                #     pt2=(width-xmax_in_cropped_image, height-ymax_in_cropped_image),
                #     color=draw_rectangle_with_color,
                #     thickness=2
                # )

        if inplace:
            super().__setattr__('cropped_image', cropped_image)
        else:
            if return_as_bbox_data_in_cropped_image:
                return BboxData(
                    image_path=self.image_path,
                    image=cropped_image,
                    xmin=xmin_in_cropped_image,
                    ymin=ymin_in_cropped_image,
                    xmax=width-xmax_in_cropped_image,
                    ymax=height-ymax_in_cropped_image,
                    label=self.label,
                    detection_score=self.detection_score,
                    classification_score=self.classification_score,
                    top_n=self.top_n,
                    labels_top_n=self.labels_top_n,
                    classification_scores_top_n=self.classification_scores_top_n
                )
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

    def asdict(self, use_special_character_func: Callable[[str], str] = None):
        return {
            'xmin': self.xmin,
            'ymin': self.ymin,
            'xmax': self.xmax,
            'ymax': self.ymax,
            'label': self.label if use_special_character_func is None else use_special_character_func(self.label),
            'top_n': self.top_n,
            'labels_top_n': self.labels_top_n,
            'additional_info': self.additional_info
        }


@dataclass(frozen=True)
class ImageData:
    image_path: Union[str, Path] = None
    image_bytes: io.BytesIO = None
    image: np.ndarray = None
    bboxes_data: List[BboxData] = None
    additional_info: Dict = field(default_factory=dict)

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
