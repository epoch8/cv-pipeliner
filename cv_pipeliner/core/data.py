import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, List, Dict, Tuple

import numpy as np
import cv2
import fsspec
from pathy import Pathy

from cv_pipeliner.utils.images import rotate_point, open_image


def open_image_for_object(
    obj: Union['ImageData', 'BboxData'],
    inplace: bool = False
) -> Union[None, np.ndarray]:
    if obj.image is not None and isinstance(obj.image, np.ndarray):
        if not inplace:
            return obj.image
        else:
            image = obj.image.copy()
    elif isinstance(obj.image, bytes) or isinstance(obj.image, str):
        image = open_image(image=obj.image, open_as_rgb=True)
    elif obj.image_path is not None:
        image = open_image(image=obj.image_path, open_as_rgb=True)
    else:
        raise ValueError("Object doesn't have any image.")

    if inplace:
        obj.image = image
    else:
        return image


@dataclass
class BboxData:
    image_path: Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO] = None
    image_name: str = None
    image: np.ndarray = None
    cropped_image: np.ndarray = None
    xmin: int = None
    ymin: int = None
    xmax: int = None
    ymax: int = None
    angle: int = 0
    detection_score: float = None
    label: str = None
    classification_score: float = None

    top_n: int = None
    labels_top_n: List[str] = None
    classification_scores_top_n: List[float] = None

    additional_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.image_path, str) or isinstance(self.image_path, Path):
            self.image_path = Pathy(self.image_path)
            self.image_name = self.image_path.name
        elif isinstance(self.image_path, fsspec.core.OpenFile):
            self.image_name = Pathy(self.image_path.path).name
        elif isinstance(self.image_path, bytes) or isinstance(self.image_path, io.BytesIO):
            self.image_name = 'bytes'
        if self.detection_score is not None:
            self.detection_score = float(self.detection_score)
        if self.classification_score is not None:
            self.classification_score = float(self.classification_score)
        if self.classification_scores_top_n is not None:
            self.classification_scores_top_n = list(map(float, self.classification_scores_top_n))

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
    ) -> Union[None, np.ndarray, 'BboxData']:

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

            points = [(self.xmin, self.ymin), (self.xmin, self.ymax), (self.xmax, self.ymin), (self.xmax, self.ymax)]
            rotated_points = [rotate_point(x=x, y=y, cx=self.xmin, cy=self.ymin, angle=self.angle) for (x, y) in points]
            xmin = max(0, min([x for (x, y) in rotated_points]))
            ymin = max(0, min([y for (x, y) in rotated_points]))
            xmax = max([x for (x, y) in rotated_points])
            ymax = max([y for (x, y) in rotated_points])
            height, width, _ = image.shape
            xmin_in_cropped_image = max(0, min(xmin_offset, xmin-xmin_offset))
            ymin_in_cropped_image = max(0, min(ymin_offset, ymin-ymin_offset))
            xmax_in_cropped_image = max(0, min(xmax_offset, width-xmax))
            ymax_in_cropped_image = max(0, min(ymax_offset, height-ymax))
            cropped_image = image[ymin-ymin_in_cropped_image:ymax+ymax_in_cropped_image,
                                  xmin-xmin_in_cropped_image:xmax+xmax_in_cropped_image]
            rotated_points_in_cropped_image = [
                [x-(xmin-xmin_in_cropped_image), y-(ymin-ymin_in_cropped_image)]
                for (x, y) in rotated_points
            ]
            if draw_rectangle_with_color is not None:
                cropped_image = cropped_image.copy()
                height, width, colors = cropped_image.shape
                rect = cv2.minAreaRect(np.array(rotated_points_in_cropped_image))
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cropped_image_zeros = np.ones_like(cropped_image)
                cv2.drawContours(
                    image=cropped_image_zeros,
                    contours=[box],
                    contourIdx=0,
                    color=draw_rectangle_with_color,
                    thickness=thickness
                )
                colored_regions = (cropped_image_zeros == draw_rectangle_with_color)
                cropped_image[colored_regions] = (
                    (1 - alpha) * cropped_image[colored_regions] + alpha * cropped_image_zeros[colored_regions]
                )

        if inplace:
            self.cropped_image = cropped_image
        else:
            if return_as_bbox_data_in_cropped_image:
                cx = rotated_points_in_cropped_image[0][0]
                cy = rotated_points_in_cropped_image[0][1]
                unrotated_points_in_cropped_image = [
                    rotate_point(x=x, y=y, cx=cx, cy=cy, angle=-self.angle)
                    for (x, y) in rotated_points_in_cropped_image
                ]
                unrotated_xmin_in_cropped_image = max(0, min([x for (x, y) in unrotated_points_in_cropped_image]))
                unrotated_ymin_in_cropped_image = max(0, min([y for (x, y) in unrotated_points_in_cropped_image]))
                unrotated_xmax_in_cropped_image = max([x for (x, y) in unrotated_points_in_cropped_image])
                unrotated_ymax_in_cropped_image = max([y for (x, y) in unrotated_points_in_cropped_image])
                return BboxData(
                    image_path=self.image_path,
                    image=cropped_image,
                    xmin=unrotated_xmin_in_cropped_image,
                    ymin=unrotated_ymin_in_cropped_image,
                    xmax=unrotated_xmax_in_cropped_image,
                    ymax=unrotated_ymax_in_cropped_image,
                    angle=self.angle,
                    label=self.label,
                    detection_score=self.detection_score,
                    classification_score=self.classification_score,
                    top_n=self.top_n,
                    labels_top_n=self.labels_top_n,
                    classification_scores_top_n=self.classification_scores_top_n,
                    additional_info={
                        **self.additional_info,
                        **{
                            'src_xmin': xmin-xmin_in_cropped_image,
                            'src_ymin': ymin-ymin_in_cropped_image
                        }
                    }
                )
            else:
                return cropped_image

    def open_image(
        self,
        inplace: bool = False
    ) -> Union[None, np.ndarray]:
        return open_image_for_object(obj=self, inplace=inplace)

    def assert_coords_are_valid(self):
        assert all(x is not None for x in [self.xmin, self.ymin, self.xmax, self.ymax])
        assert self.xmin <= self.xmax and self.ymin <= self.ymax

    def assert_label_is_valid(self):
        assert self.label is not None

    def asdict(self) -> Dict:
        if isinstance(self.image_path, fsspec.core.OpenFile):
            protocol = self.image_path.fs.protocol[0]
            image_path_str = f"{protocol}://{str(self.image_path.path)}"
        else:
            image_path_str = str(self.image_path) if self.image_path is not None else None
        image_str = self.image if isinstance(self.image, str) else None
        return {
            'image_path': image_path_str,
            'image': image_str,
            'xmin': int(self.xmin),
            'ymin': int(self.ymin),
            'xmax': int(self.xmax),
            'ymax': int(self.ymax),
            'angle': int(self.angle),
            'label': str(self.label),
            'top_n': int(self.top_n) if self.top_n is not None else None,
            'labels_top_n': [str(label) for label in self.labels_top_n] if self.labels_top_n is not None else None,
            'classification_scores_top_n': [
                str(round(score, 3)) for score in self.classification_scores_top_n
            ] if self.classification_scores_top_n is not None else None,
            'detection_score': str(round(self.detection_score, 3)) if self.detection_score is not None else None,
            'classification_score': str(
                round(self.classification_score, 3
            )) if self.classification_score is not None else None,
            'additional_info': self.additional_info
        }

    def _from_dict(self, d):
        for key in [
            'image_path', 'image', 'xmin', 'ymin', 'xmax', 'ymax',
            'angle', 'label', 'top_n', 'labels_top_n', 'classification_scores_top_n',
            'detection_score', 'classification_score',
            'additional_info',
        ]:
            if key in d:
                super().__setattr__(key, d[key])
        self.__post_init__()

        return self

    @staticmethod
    def from_dict(d):
        return BboxData()._from_dict(d)


@dataclass
class ImageData:
    image_path: Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO] = None
    image_name: str = None
    image: np.ndarray = None
    bboxes_data: List[BboxData] = field(default_factory=list)
    additional_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.image_path, str) or isinstance(self.image_path, Path):
            self.image_path = Pathy(self.image_path)
            self.image_name = self.image_path.name
        elif isinstance(self.image_path, fsspec.core.OpenFile):
            self.image_name = Pathy(self.image_path.path).name
        elif isinstance(self.image_path, bytes) or isinstance(self.image_path, io.BytesIO):
            self.image_name = 'bytes'

    def open_image(
        self,
        inplace: bool = False
    ) -> Union[None, np.ndarray]:
        return open_image_for_object(obj=self, inplace=inplace)

    def asdict(self) -> Dict:
        if isinstance(self.image_path, fsspec.core.OpenFile):
            protocol = self.image_path.fs.protocol[0]
            image_path_str = f"{protocol}://{str(self.image_path.path)}"
        else:
            image_path_str = str(self.image_path) if self.image_path is not None else None
        image_str = self.image if isinstance(self.image, str) else None
        return {
            'image_path': image_path_str,
            'image': image_str,
            'bboxes_data': [bbox_data.asdict() for bbox_data in self.bboxes_data],
            'additional_info': self.additional_info
        }

    def _from_dict(self, d):
        for key in ['image_path', 'image', 'additional_info']:
            if key in d:
                super().__setattr__(key, d[key])
        if 'bboxes_data' in d:
            bboxes_data = [BboxData() for i in range(len(d['bboxes_data']))]
            for bbox_data, d_i in zip(bboxes_data, d['bboxes_data']):
                bbox_data._from_dict(d_i)
            self.bboxes_data = bboxes_data
        self.__post_init__()

        return self

    @staticmethod
    def from_dict(d):
        return ImageData()._from_dict(d)
