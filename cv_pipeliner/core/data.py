import copy
import io
import json

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Union, List, Dict, Tuple, Optional, Type

import numpy as np
import fsspec
import PIL

from pathy import Pathy

from cv_pipeliner.utils.images import is_base64, open_image
from cv_pipeliner.utils.imagesize import get_image_size


def open_image_for_object(
    obj: Union['ImageData', 'BboxData'],
    inplace: bool = False,
    returns_none_if_empty: bool = False
) -> Optional[np.ndarray]:
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
        if returns_none_if_empty:
            return None
        raise ValueError("Object doesn't have any image.")

    if inplace:
        obj.image = image
    else:
        return image


ImagePath = Optional[Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image]]


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
    image: np.ndarray = field(default=None, repr=False)
    cropped_image: np.ndarray = field(default=None, repr=False)
    xmin: Union[int, float] = None
    ymin: Union[int, float] = None
    xmax: Union[int, float] = None
    ymax: Union[int, float] = None
    keypoints: List[Tuple[int, int]] = field(default_factory=list)

    detection_score: float = field(default=None, repr=False)
    label: str = None
    classification_score: float = field(default=None, repr=False)
    top_n: int = field(default=None, repr=False)
    labels_top_n: List[str] = field(default=None, repr=False)
    classification_scores_top_n: List[float] = field(default=None, repr=False)

    additional_bboxes_data: List['BboxData'] = field(default_factory=list)
    additional_info: Dict = field(default_factory=dict)

    meta_width: int = None
    meta_height: int = None

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

        if self.image is not None:
            self.image = np.array(self.image)
            self.meta_width, self.meta_height = self.get_image_size()

    @property
    def image_name(self) -> str:
        return get_image_name(self.image_path)

    @property
    def coords(self) -> Tuple[int, int, int, int]:
        return (round(self.xmin), round(self.ymin), round(self.xmax), round(self.ymax))

    @property
    def coords_n(self) -> Tuple[float, float, float, float]:
        width, height = self.get_image_size()
        return self.xmin / width, self.ymin / height, self.xmax / width, self.ymax / height
    
    @property
    def keypoints_n(self) -> List[Tuple[float]]:
        width, height = self.get_image_size()
        keypoints = self.keypoints.copy().astype(float).reshape((-1, 2))
        keypoints[:, 0] /= width
        keypoints[:, 1] /= height
        return keypoints

    @property
    def area(self) -> int:
        return (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)

    def coords_with_offset(
        self,
        xmin_offset: Union[int, float] = 0,
        ymin_offset: Union[int, float] = 0,
        xmax_offset: Union[int, float] = 0,
        ymax_offset: Union[int, float] = 0,
        source_image: np.ndarray = None
    ) -> Tuple[int, int, int, int]:
        if source_image is not None:
            height, width = source_image.shape[0], source_image.shape[1]
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
            max(0, round(self.xmin-xmin_in_cropped_image)),
            max(0, round(self.ymin-ymin_in_cropped_image)),
            min(width-1, round(self.xmax+xmax_in_cropped_image)),
            min(height-1, round(self.ymax+ymax_in_cropped_image))
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
        image_data_cls: Type['ImageData'] = 'ImageData',
    ) -> Optional[Union[np.ndarray, 'BboxData']]:
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
            self.meta_height, self.meta_width = image.shape[0:2]

            assert self.xmin < self.xmax and self.ymin < self.ymax

            xmin, ymin, xmax, ymax = self.coords_with_offset(
                xmin_offset, ymin_offset, xmax_offset, ymax_offset, source_image
            )
            cropped_image = image[ymin:ymax, xmin:xmax] if image is not None else None
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

                if image_data_cls == 'ImageData':
                    from cv_pipeliner.core.data import ImageData
                    image_data_cls = ImageData
                image_data = image_data_cls(
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
        inplace: bool = False,
        returns_none_if_empty: bool = False
    ) -> Optional[np.ndarray]:
        image = open_image_for_object(obj=self, inplace=inplace, returns_none_if_empty=returns_none_if_empty)
        if image is not None:
            self.meta_height, self.meta_width = image.shape[0:2]
        return image

    def get_image_size(self, force_update_meta: bool = False) -> Tuple[int, int]:
        """
            Returns (width, height) of image without opening it fully.
        """
        if self.meta_height is None or self.meta_width is None or force_update_meta:
            if self.image is not None:
                self.meta_height, self.meta_width = self.image.shape[0:2]
            else:
                self.meta_width, self.meta_height = get_image_size(self.image_path)
        if self.image is not None:
            self.meta_height, self.meta_width = self.image.shape[0:2]
        return self.meta_width, self.meta_height

    def get_image_size_without_exif_tag(self, exif_transpose: bool = True) -> Tuple[int, int]:
        """
            Returns (width, height) of image without opening it fully.
        """
        if self.image is None:
            return get_image_size(self.image_path, exif_transpose=False)
        if self.image is not None:
            meta_height, meta_width = self.image.shape[0:2]
            return meta_height, meta_width
        return (None, None)

    def json(self, include_image_path: bool = True, force_include_meta: bool = False) -> Dict:
        result_json = {
            'xmin': int(self.xmin) if isinstance(self.xmin, (int, np.int64)) else round(self.xmin, 6),
            'ymin': int(self.ymin) if isinstance(self.ymin, (int, np.int64)) else round(self.ymin, 6),
            'xmax': int(self.xmax) if isinstance(self.xmax, (int, np.int64)) else round(self.xmax, 6),
            'ymax': int(self.ymax) if isinstance(self.ymax, (int, np.int64)) else round(self.ymax, 6),
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

        if force_include_meta:
            self.get_image_size()  # write meta inplace if empty
        if self.meta_width is not None:
            result_json['meta_width'] = int(self.meta_width)
        if self.meta_height is not None:
            result_json['meta_height'] = int(self.meta_height)

        return result_json

    def _from_json(
        self, d: Dict,
        image_path: ImagePath = None,
    ):
        for key in self.__dataclass_fields__:
            if key in d:
                if key == 'additional_bboxes_data':
                    d[key] = [
                        type(self).from_json(bbox_data, image_path=image_path, bbox_data_cls=type(self))
                        for bbox_data in d['additional_bboxes_data']
                    ]
                super().__setattr__(key, d[key])

        if image_path is not None:
            self.image_path = image_path

        self.__post_init__()

        return self

    @staticmethod
    def from_json(
        d: Optional[Union[Path, str, Dict[str, Any], fsspec.core.OpenFile]],
        image_path: ImagePath = None,
        bbox_data_cls: Type['BboxData'] = None,
        **kwargs
    ):
        if bbox_data_cls is None:
            bbox_data_cls = BboxData
        if d is None:
            return bbox_data_cls(image_path=image_path)
        if isinstance(d, str) or isinstance(d, Path):
            with fsspec.open(d, 'r') as f:
                d = json.loads(f.read())
        elif isinstance(d, fsspec.core.OpenFile):
            with d as f:
                d = json.load(f)
        return bbox_data_cls(**kwargs)._from_json(d=d, image_path=image_path)

    def count_children(self) -> int:
        global counting
        counting = 0

        def counts(bbox_data: BboxData):
            global counting
            counting += len(bbox_data.additional_bboxes_data)
            for additional_bbox_data in bbox_data.additional_bboxes_data:
                counts(additional_bbox_data)

        for additional_bbox_data in self.additional_bboxes_data:
            counts(additional_bbox_data)

        return counting


@dataclass
class ImageData:
    image_path: ImagePath = None
    image: np.ndarray = field(default=None, repr=False)
    bboxes_data: List[BboxData] = field(default_factory=list)

    label: str = None
    keypoints: List[Tuple[int, int]] = field(default_factory=list)
    additional_info: Dict = field(default_factory=dict)
    classification_score: float = field(default=None, repr=False)
    top_n: int = field(default=None, repr=False)
    labels_top_n: List[str] = field(default=None, repr=False)
    classification_scores_top_n: List[float] = field(default=None, repr=False)

    meta_width: int = None
    meta_height: int = None

    def __post_init__(self):
        if isinstance(self.image_path, Path) or (isinstance(self.image_path, str) and not is_base64(self.image_path)):
            self.image_path = Pathy(self.image_path)

        self.keypoints = np.array(self.keypoints).astype(int).reshape((-1, 2))

        # Apply these to all bboxes_data (look __setattr__)
        self.image_path = self.image_path
        self.image = self.image
        if self.classification_score is not None:
            self.classification_score = float(self.classification_score)
        if self.classification_scores_top_n is not None:
            self.classification_scores_top_n = list(map(float, self.classification_scores_top_n))

        if self.image is not None:
            self.image = np.array(self.image)
            self.get_image_size(force_update_meta=True)
        if self.meta_height is None and self.meta_width is None and len(self.bboxes_data) > 0:
            for bbox_data in self.bboxes_data:
                if bbox_data.meta_height is not None and bbox_data.meta_width is not None:
                    self.meta_height, self.meta_width = bbox_data.meta_height, bbox_data.meta_width

    @property
    def image_name(self):
        return get_image_name(self.image_path)

    @property
    def keypoints_n(self) -> List[Tuple[float]]:
        width, height = self.get_image_size()
        keypoints = self.keypoints.copy().astype(float).reshape((-1, 2))
        keypoints[:, 0] /= width
        keypoints[:, 1] /= height
        return keypoints

    def open_image(
        self,
        inplace: bool = False,
        returns_none_if_empty: bool = False
    ) -> Optional[np.ndarray]:
        image = open_image_for_object(obj=self, inplace=inplace, returns_none_if_empty=returns_none_if_empty)
        if image is not None:
            self.meta_height, self.meta_width = image.shape[0:2]

        return image

    def get_image_size(self, force_update_meta: bool = False) -> Tuple[int, int]:
        """
            Returns (width, height) of image without opening it fully.
        """
        if self.meta_height is None or self.meta_width is None or force_update_meta:
            if self.image is not None:
                self.meta_height, self.meta_width = self.image.shape[0:2]
            else:
                self.meta_width, self.meta_height = get_image_size(self.image_path)
        if self.image is not None:
            self.meta_height, self.meta_width = self.image.shape[0:2]
        return self.meta_width, self.meta_height

    def get_image_size_without_exif_tag(self, exif_transpose: bool = True) -> Tuple[int, int]:
        """
            Returns (width, height) of image without opening it fully.
        """
        if self.image is None:
            return get_image_size(self.image_path, exif_transpose=False)
        if self.image is not None:
            meta_height, meta_width = self.image.shape[0:2]
            return meta_height, meta_width
        return (None, None)

    def json(self, force_include_meta: bool = False) -> Dict:
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
        if force_include_meta:
            self.get_image_size()  # write meta inplace
        if self.classification_score is not None:
            result_json['classification_score'] = str(round(self.classification_score, 3))
        if self.top_n is not None:
            result_json['top_n'] = int(self.top_n)
        if self.labels_top_n is not None:
            result_json['labels_top_n'] = [str(label) for label in self.labels_top_n]
        if self.classification_scores_top_n is not None:
            result_json['classification_scores_top_n'] = [
                str(round(score, 3)) for score in self.classification_scores_top_n
            ]
        if self.meta_width is not None:
            result_json['meta_width'] = int(self.meta_width)
        if self.meta_height is not None:
            result_json['meta_height'] = int(self.meta_height)

        return result_json

    def _from_json(
        self,
        d: Union[str, Path, Dict],
        bbox_data_cls: Type[BboxData] = None
    ):
        for key in self.__dataclass_fields__:
            if key in d:
                if key == 'image_path' and self.image_path is not None:
                    continue
                if key == 'bboxes_data':
                    d[key] = [
                        bbox_data_cls.from_json(bbox_data, image_path=self.image_path, bbox_data_cls=bbox_data_cls)
                        for bbox_data in d['bboxes_data']
                    ]
                super().__setattr__(key, d[key])

        self.__post_init__()

        return self

    @staticmethod
    def from_json(
        d: Optional[Union[Path, str, Dict[str, Any], fsspec.core.OpenFile]],
        image_path: ImagePath = None,
        image_data_cls: Type['ImageData'] = 'ImageData',
        bbox_data_cls: Type[BboxData] = BboxData,
        **kwargs
    ):
        if image_data_cls is None or image_data_cls == 'ImageData':
            image_data_cls = ImageData
        if d is None:
            return image_data_cls(image_path=image_path)
        if isinstance(d, str) or isinstance(d, Path):
            with fsspec.open(d, 'r') as f:
                d = json.loads(f.read())
        elif isinstance(d, fsspec.core.OpenFile):
            with d as f:
                d = json.load(f)
        if 'image_data' in d:
            d = d['image_data']
        return image_data_cls(image_path=image_path, **kwargs)._from_json(d, bbox_data_cls)

    def is_empty(self):
        return self.image_path is None and self.image is None

    def __setattr__(
        self, name: str, value: Any,
        apply_to_bboxes_data: bool = True,
        force_update_meta: bool = False
    ) -> None:

        if hasattr(self, name) and (
            (name == 'image' and np.array_equal(self.image, value)) or
            (name == 'image_path' and str(self.image_path) != str(value))
        ):
            force_update_meta = True

        super().__setattr__(name, value)
        if name in ['image_path', 'image', 'meta_width', 'meta_height']:
            if name == 'image' and isinstance(value, np.ndarray) and force_update_meta:  # imagesize possible is changed
                self.get_image_size(force_update_meta=force_update_meta)

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

    def count_children_bboxes_data(self) -> int:
        count = 0
        for bbox_data in self.bboxes_data:
            count += (1 + bbox_data.count_children())
        return count
