import copy
import io
import json

from pathlib import Path
from pydantic import BaseModel, Field, validator, root_validator
from typing import Any, Union, List, Dict, Tuple, Optional, Type

import numpy as np
import fsspec
import PIL

from pathy import Pathy

from cv_pipeliner.utils.images import is_base64, open_image
from cv_pipeliner.utils.imagesize import get_image_size


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


def get_meta_image_size(
    image_path: Optional[Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image]],
    image: Optional[np.ndarray],
    meta_height: Optional[int],
    meta_width: Optional[int],
    force_update_meta: bool = False
):
    """
        Returns (width, height) of image without opening it fully.
    """
    if meta_height is None or meta_width is None or force_update_meta:
        if image is not None:
            meta_height, meta_width = image.shape[0:2]
        else:
            meta_width, meta_height = get_image_size(image_path)
    if image is not None:
        meta_height, meta_width = image.shape[0:2]
    return meta_width, meta_height


class BaseImageData(BaseModel):
    image_path: Optional[Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image]] = None
    image: Optional[np.ndarray] = Field(default=None, repr=False, exclude=True)
    label: Optional[str] = None
    keypoints: np.ndarray = Field(default_factory=lambda: np.array([]).astype(int).reshape((-1, 2)))
    detection_score: Optional[float] = Field(default=None, repr=False)
    classification_score: Optional[float] = Field(default=None, repr=False)
    top_n: Optional[int] = Field(default=None, repr=False)
    labels_top_n: Optional[Union[List[List[str]], np.ndarray, List[np.ndarray], List[str]]] = Field(
        default=None, repr=False
    )
    classification_scores_top_n: Optional[Union[List[List[float]], np.ndarray, List[np.ndarray], List[float]]] = Field(
        default=None, repr=False
    )
    additional_info: Dict[str, Any] = Field(default_factory=dict)
    meta_width: Optional[int] = None
    meta_height: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        json_encoders = {
            Optional[Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image]]: get_image_path_as_str,
            np.ndarray: lambda x: x.tolist()
        }
        smart_union = True

    @validator('image_path', pre=True)
    def parse_image_path(cls, image_path):
        if isinstance(image_path, Path) or (isinstance(image_path, str) and not is_base64(image_path)):
            image_path = Pathy(image_path)
        return image_path

    @validator('keypoints', pre=True)
    def parse_keypoints(cls, keypoints):
        if keypoints is None:
            keypoints = []
        return np.array(keypoints).astype(int).reshape((-1, 2))

    @validator('image', pre=True)
    def parse_image(cls, image, values):
        if image is not None:
            image = np.array(image)
        return image

    @root_validator(pre=False)
    def set_fields(cls, values: dict) -> dict:
        if values['image'] is not None:
            values['meta_width'], values['meta_height'] = get_meta_image_size(
                image_path=values['image_path'],
                image=values['image'],
                meta_height=values['meta_width'],
                meta_width=values['meta_width'],
                force_update_meta=True
            )
        return values

    def get_image_size(self, force_update_meta: bool = False) -> Tuple[int, int]:
        """
            Returns (width, height) of image without opening it fully.
        """
        self.meta_width, self.meta_height = get_meta_image_size(
            image_path=self.image_path,
            image=self.image,
            meta_height=self.meta_height,
            meta_width=self.meta_width,
            force_update_meta=force_update_meta
        )
        return self.meta_width, self.meta_height

    @property
    def image_name(self):
        return get_image_name(self.image_path)

    @property
    def keypoints_n(self) -> List[Tuple[float]]:
        width, height = self.get_image_size()
        keypoints = self.keypoints.astype(float)
        keypoints[:, 0] /= width
        keypoints[:, 1] /= height
        return keypoints

    def open_image(
        self,
        inplace: bool = False,
        returns_none_if_empty: bool = False
    ) -> Optional[np.ndarray]:
        if self.image is not None and isinstance(self.image, np.ndarray):
            if not inplace:
                return self.image
            else:
                image = self.image.copy()
        elif isinstance(self.image, bytes) or isinstance(self.image, str):
            image = open_image(image=self.image, open_as_rgb=True)
        elif self.image_path is not None:
            image = open_image(image=self.image_path, open_as_rgb=True)
        else:
            if returns_none_if_empty:
                return None
            raise ValueError("Object doesn't have any image.")

        if inplace:
            self.image = image

        if image is not None:
            self.get_image_size(force_update_meta=True)

        return image

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

    def json(
        self,
        include_image_path: bool = True,
        force_include_meta: bool = False,
        **kwargs
    ) -> str:
        kwargs = kwargs.copy()
        exclude = kwargs.pop('exclude', set())
        if force_include_meta:
            self.get_image_size()  # write meta inplace if empty
        if not include_image_path:
            exclude.add('image_path')
        return super().json(
            exclude=exclude,
            exclude_none=kwargs.pop('exclude_none', True),
            exclude_unset=kwargs.pop('exclude_unset', True),
            **kwargs
        )

    @classmethod
    def from_json(
        cls,
        d: Optional[Union[Path, str, Dict[str, Any], fsspec.core.OpenFile]],
        image_path: Optional[Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image]] = None,
    ):
        if cls is None:
            cls = BaseImageData

        if d is None:
            return cls(image_path=image_path)
        if isinstance(d, (str, Path)):
            try:
                d = json.loads(str(d))
            except Exception:
                with fsspec.open(d, 'r') as f:
                    d = json.loads(f.read())
        elif isinstance(d, fsspec.core.OpenFile):
            with d as f:
                d = json.load(f)
        if image_path is not None:
            d['image_path'] = image_path

        return cls(**d)

    def is_empty(self):
        return self.image_path is None and self.image is None


class BboxData(BaseImageData):
    xmin: Optional[Union[int, float]] = None
    ymin: Optional[Union[int, float]] = None
    xmax: Optional[Union[int, float]] = None
    ymax: Optional[Union[int, float]] = None
    cropped_image: Optional[np.ndarray] = Field(default=None, repr=False)
    additional_bboxes_data: Optional[List['BboxData']] = Field(default_factory=list)

    @validator('xmin', 'ymin', pre=True)
    def parse_xmin(cls, v):
        if v is not None:
            v = max(0, v)
        return v

    @property
    def coords(self) -> Tuple[int, int, int, int]:
        return (round(self.xmin), round(self.ymin), round(self.xmax), round(self.ymax))

    @property
    def coords_n(self) -> Tuple[float, float, float, float]:
        width, height = self.get_image_size()
        return self.xmin / width, self.ymin / height, self.xmax / width, self.ymax / height

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


class ImageData(BaseImageData):
    bboxes_data: List[BboxData] = Field(default_factory=list)

    def __setattr__(
        self, name: str, value: Any,
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
