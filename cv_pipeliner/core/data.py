import copy
import io
import json

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Union, List, Dict, Tuple

import numpy as np
import fsspec
import PIL

from pathy import Pathy

from cv_pipeliner.utils.images import is_base64, open_image
from cv_pipeliner.utils.imagesize import get_image_size


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


ImagePath = Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image]


def get_image_name(image_path) -> str:
    if isinstance(image_path, Pathy):
        return image_path.name
    elif isinstance(image_path, fsspec.core.OpenFile):
        return Pathy(image_path.path).name
    elif isinstance(image_path, str) or isinstance(image_path, bytes) or isinstance(image_path, io.BytesIO):
        return 'bytes'
    elif isinstance(image_path, PIL.Image.Image):
        return 'PIL.Image.Image'


def get_image_path_as_str(image_path) -> str:
    if isinstance(image_path, fsspec.core.OpenFile):
        protocol = image_path.fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[0]
        prefix = f"{protocol}://"
        if protocol == 'file':
            prefix = ''
        image_path_str = f"{prefix}{str(image_path.path)}"
    else:
        image_path_str = str(image_path) if (
            image_path is not None and not isinstance(image_path, PIL.Image.Image)
        ) else None

    return image_path_str


@dataclass
class BboxData:
    image_path: ImagePath = None
    image: np.ndarray = None
    cropped_image: np.ndarray = None
    xmin: Union[int, float] = None
    ymin: Union[int, float] = None
    xmax: Union[int, float] = None
    ymax: Union[int, float] = None
    keypoints: List[Tuple[int, int]] = field(default_factory=list)

    detection_score: float = None
    label: str = None
    classification_score: float = None
    top_n: int = None
    labels_top_n: List[str] = None
    classification_scores_top_n: List[float] = None

    additional_bboxes_data: List['BboxData'] = field(default_factory=list)
    additional_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.image_path, Path) or (isinstance(self.image_path, str) and not is_base64(self.image_path)):
            self.image_path = Pathy(self.image_path)

        if self.xmin is not None:
            self.xmin = max(0, self.xmin)
        if self.ymin is not None:
            self.ymin = max(0, self.ymin)

        if self.detection_score is not None:
            self.detection_score = float(self.detection_score)
        if self.classification_score is not None:
            self.classification_score = float(self.classification_score)
        if self.classification_scores_top_n is not None:
            self.classification_scores_top_n = list(map(float, self.classification_scores_top_n))

        self.keypoints = np.array(self.keypoints).astype(int).reshape((-1, 2))

    @property
    def image_name(self) -> str:
        return get_image_name(self.image_path)

    @property
    def coords(self) -> Tuple[int, int, int, int]:
        return (round(self.xmin), round(self.ymin), round(self.xmax), round(self.ymax))

    @property
    def coords_n(self) -> Tuple[int, int, int, int]:
        width, height = self.get_image_size()
        return self.xmin / width, self.ymin / height, self.xmax / width, self.ymax / height

    def coords_with_offset(
        self,
        xmin_offset: Union[int, float] = 0,
        ymin_offset: Union[int, float] = 0,
        xmax_offset: Union[int, float] = 0,
        ymax_offset: Union[int, float] = 0,
        source_image: np.ndarray = None
    ) -> Tuple[int, int, int, int]:
        if source_image is not None:
            height, width, _ = source_image.shape
        else:
            width, height = self.get_image_size()
        if isinstance(xmin_offset, float):
            assert 0 < xmin_offset and xmin_offset < 1
            xmin_offset = int((self.xmax - self.xmin) * xmin_offset)
        if isinstance(ymin_offset, float):
            assert 0 < ymin_offset and ymin_offset < 1
            ymin_offset = int((self.ymax - self.ymin) * ymin_offset)
        if isinstance(xmax_offset, float):
            assert 0 < xmax_offset and xmax_offset < 1
            xmax_offset = int((self.xmax - self.xmin) * xmax_offset)
        if isinstance(ymax_offset, float):
            assert 0 < ymax_offset and ymax_offset < 1
            ymax_offset = int((self.ymax - self.ymin) * ymax_offset)
        xmin_in_cropped_image = max(0, min(xmin_offset, self.xmin-xmin_offset))
        ymin_in_cropped_image = max(0, min(ymin_offset, self.ymin-ymin_offset))
        xmax_in_cropped_image = max(0, min(xmax_offset, width-self.xmax))
        ymax_in_cropped_image = max(0, min(ymax_offset, height-self.ymax))
        return (
            round(self.xmin-xmin_in_cropped_image),
            round(self.ymin-ymin_in_cropped_image),
            round(self.xmax+xmax_in_cropped_image),
            round(self.ymax+ymax_in_cropped_image)
        )

    def open_cropped_image(
        self,
        inplace: bool = False,
        source_image: np.ndarray = None,
        xmin_offset: Union[int, float] = 0,
        ymin_offset: Union[int, float] = 0,
        xmax_offset: Union[int, float] = 0,
        ymax_offset: Union[int, float] = 0,
        return_as_image_data: bool = False,
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

            xmin, ymin, xmax, ymax = self.coords_with_offset(xmin_offset, ymin_offset, xmax_offset, ymax_offset, source_image)
            cropped_image = image[ymin:ymax, xmin:xmax]

        if inplace:
            self.cropped_image = cropped_image
        else:
            if return_as_image_data:
                keypoints = self.keypoints.copy()
                keypoints[:, 0] -= xmin
                keypoints[:, 1] -= ymin
                additional_bboxes_data = copy.deepcopy(self.additional_bboxes_data)

                def crop_additional_bbox_data(bbox_data: BboxData):
                    bbox_data.image_path = None
                    bbox_data.image = cropped_image
                    bbox_data.xmin -= xmin
                    bbox_data.ymin -= ymin
                    bbox_data.xmax -= xmin
                    bbox_data.ymax -= ymin
                    for additional_bbox_data in bbox_data.additional_bboxes_data:
                        crop_additional_bbox_data(additional_bbox_data)
                for additional_bbox_data in additional_bboxes_data:
                    crop_additional_bbox_data(additional_bbox_data)

                from cv_pipeliner.core.data import ImageData
                image_data = ImageData(
                    image=cropped_image,
                    bboxes_data=additional_bboxes_data,
                    label=self.label,
                    keypoints=keypoints,
                    additional_info=self.additional_info
                )
                return image_data
            else:
                return cropped_image

    def open_image(
        self,
        inplace: bool = False
    ) -> Union[None, np.ndarray]:
        return open_image_for_object(obj=self, inplace=inplace)

    def get_image_size(self) -> Tuple[int, int]:
        """
            Returns (width, height) of image without opening it fully.
        """
        if self.image is not None:
            height, width, _ = self.image.shape
        else:
            width, height = get_image_size(self.image_path)
        return width, height

    def json(self, include_image_path: bool = True) -> Dict:
        result_json = {
            'xmin': self.xmin if isinstance(self.xmin, int) else round(self.xmin, 6),
            'ymin': self.ymin if isinstance(self.ymin, int) else round(self.ymin, 6),
            'xmax': self.xmax if isinstance(self.xmax, int) else round(self.xmax, 6),
            'ymax': self.ymax if isinstance(self.ymax, int) else round(self.ymax, 6),
        }
        if include_image_path:
            result_json['image_path'] = get_image_path_as_str(self.image_path)
        if self.label is not None:
            result_json['label'] = str(self.label)
        if len(self.keypoints) > 0:
            result_json['keypoints'] = np.array(self.keypoints).astype(int).tolist()
        if self.top_n is not None:
            result_json['top_n'] = int(self.top_n)
        if self.labels_top_n is not None:
            result_json['labels_top_n'] = [str(label) for label in self.labels_top_n]
        if self.classification_scores_top_n is not None:
            result_json['classification_scores_top_n'] = [
                str(round(score, 3)) for score in self.classification_scores_top_n
            ]
        if self.detection_score is not None:
            result_json['detection_score'] = str(round(self.detection_score, 3))
        if self.classification_score is not None:
            result_json['classification_score'] = str(round(self.classification_score, 3))
        if len(self.additional_bboxes_data) > 0:
            result_json['additional_bboxes_data'] = [
                bbox_data.json(include_image_path=include_image_path) for bbox_data in self.additional_bboxes_data
            ]
        if len(self.additional_info) > 0:
            result_json['additional_info'] = self.additional_info

        return result_json

    def _from_json(self, d: Dict, image_path: ImagePath = None):
        for key in BboxData.__dataclass_fields__:
            if key in d:
                if key == 'additional_bboxes_data':
                    d[key] = [
                        BboxData.from_json(bbox_data, image_path=image_path)
                        for bbox_data in d['additional_bboxes_data']
                    ]
                super().__setattr__(key, d[key])

        if image_path is not None:
            self.image_path = image_path

        self.__post_init__()

        return self

    @staticmethod
    def from_json(d: Union[None, Path, str, Dict[str, Any], fsspec.core.OpenFile], image_path: ImagePath = None):
        if d is None:
            return BboxData(image_path=image_path)
        if isinstance(d, str) or isinstance(d, Path):
            with fsspec.open(d, 'r') as f:
                d = json.loads(f.read())
        elif isinstance(d, fsspec.core.OpenFile):
            with d as f:
                d = json.load(f)

        return BboxData()._from_json(d=d, image_path=image_path)


@dataclass
class ImageData:
    image_path: ImagePath = None
    image: np.ndarray = None
    bboxes_data: List[BboxData] = field(default_factory=list)

    label: str = None
    keypoints: List[Tuple[int, int]] = field(default_factory=list)
    additional_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.image_path, Path) or (isinstance(self.image_path, str) and not is_base64(self.image_path)):
            self.image_path = Pathy(self.image_path)

        self.keypoints = np.array(self.keypoints).astype(int).reshape((-1, 2))

        # Apply these to all bboxes_data (look __setattr__)
        self.image_path = self.image_path
        self.image = self.image

        if self.image is not None:
            self.image = np.array(self.image)

    @property
    def image_name(self):
        return get_image_name(self.image_path)

    def open_image(
        self,
        inplace: bool = False
    ) -> Union[None, np.ndarray]:
        return open_image_for_object(obj=self, inplace=inplace)

    def get_image_size(self) -> Tuple[int, int]:
        """
            Returns (width, height) of image without opening it fully.
        """
        if self.image is not None:
            height, width, _ = self.image.shape
        else:
            width, height = get_image_size(self.image_path)
        return width, height

    def json(self) -> Dict:
        result_json = {
            'image_path': get_image_path_as_str(self.image_path),
            'bboxes_data': [bbox_data.json(include_image_path=False) for bbox_data in self.bboxes_data],
        }
        if self.label is not None:
            result_json['label'] = self.label
        if len(self.keypoints) > 0:
            result_json['keypoints'] = np.array(self.keypoints).astype(int).tolist()
        if len(self.additional_info) > 0:
            result_json['additional_info'] = self.additional_info

        return result_json

    def _from_json(self, d: Union[str, Path, Dict]):
        for key in ImageData.__dataclass_fields__:
            if key in d:
                if key == 'image_path' and self.image_path is not None:
                    continue
                if key == 'bboxes_data':
                    d[key] = [
                        BboxData.from_json(bbox_data, image_path=self.image_path)
                        for bbox_data in d['bboxes_data']
                    ]
                super().__setattr__(key, d[key])

        self.__post_init__()

        return self

    @staticmethod
    def from_json(d: Union[None, Path, str, Dict[str, Any], fsspec.core.OpenFile], image_path: ImagePath = None):
        if d is None:
            return ImageData(image_path=image_path)
        if isinstance(d, str) or isinstance(d, Path):
            with fsspec.open(d, 'r') as f:
                d = json.loads(f.read())
        elif isinstance(d, fsspec.core.OpenFile):
            with d as f:
                d = json.load(f)
        if 'image_data' in d:
            d = d['image_data']
        return ImageData(image_path=image_path)._from_json(d)

    def is_empty(self):
        return self.image_path is None and self.image is None

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == 'image_path' or name == 'image':
            def change_images_in_bbox_data(bbox_data: BboxData):
                bbox_data.__setattr__(name, value)
                for additional_bbox_data in bbox_data.additional_bboxes_data:
                    change_images_in_bbox_data(additional_bbox_data)

            if hasattr(self, 'bboxes_data'):
                for bbox_data in self.bboxes_data:
                    change_images_in_bbox_data(bbox_data)

    def find_bbox_data_by_coords(self, xmin: int, ymin: int, xmax: int, ymax: int) -> BboxData:
        bboxes_data_coords = [bbox_data.coords for bbox_data in self.bboxes_data]
        return self.bboxes_data[bboxes_data_coords.index((xmin, ymin, xmax, ymax))]
