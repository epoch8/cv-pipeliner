import copy
import io
import json
from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import fsspec
import numpy as np
import PIL
from pathy import Pathy

from cv_pipeliner.utils.images import is_base64, open_image
from cv_pipeliner.utils.imagesize import get_image_size


def get_image_name(image_path) -> str:
    if isinstance(image_path, Pathy):
        return image_path.name
    elif isinstance(image_path, fsspec.core.OpenFile):
        return Pathy.fluid(image_path.path).name
    elif isinstance(image_path, str) or isinstance(image_path, bytes) or isinstance(image_path, io.BytesIO):
        return "bytes"
    elif isinstance(image_path, PIL.Image.Image):
        return "PIL.Image.Image"


def get_image_path_as_str(image_path) -> str:
    if isinstance(image_path, fsspec.core.OpenFile):
        protocol = image_path.fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[0]
        prefix = f"{protocol}://"
        if protocol == "file":
            prefix = ""
        image_path_str = f"{prefix}{str(image_path.path)}"
    else:
        image_path_str = (
            str(image_path) if (image_path is not None and not isinstance(image_path, PIL.Image.Image)) else None
        )

    return image_path_str


def get_meta_image_size(
    image_path: Optional[Union[str, Path, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image]],
    image: Optional[np.ndarray],
    meta_height: Optional[int],
    meta_width: Optional[int],
    exif_transpose: bool = False,
):
    """
    Returns (width, height) of image without opening it fully.
    """
    if image is not None:
        meta_height, meta_width = image.shape[0:2]
    elif exif_transpose and image_path is not None:
        meta_width, meta_height = get_image_size(image_path, exif_transpose=exif_transpose)
    else:
        if meta_height is None or meta_width is None:
            if image_path is None:
                raise ValueError("(get_meta_image_size) Fields image_path or image are None")
            meta_width, meta_height = get_image_size(image_path, exif_transpose=exif_transpose)
    return meta_width, meta_height


class BaseImageData(BaseModel):
    _SOURCE_FIELDS: ClassVar[Tuple[str, ...]] = ("image_path", "image", "meta_width", "meta_height")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    image_path: Optional[Union[str, Path, Pathy, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image]] = None
    image: Optional[np.ndarray] = Field(default=None, repr=False, exclude=True)
    label: Optional[str] = None
    keypoints: np.ndarray = Field(default_factory=lambda: np.array([]).astype(int).reshape((-1, 2)))
    mask: Union[
        Union[str, Path, Pathy, fsspec.core.OpenFile, bytes, io.BytesIO, PIL.Image.Image],  # path to mask image
        np.ndarray,  # mask image
        List[List[int]],  # [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...], ...]
        List[List[Tuple[int, int]]],  # [[(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...]
        List[np.ndarray],  # [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...], ...]
    ] = Field(default_factory=lambda: [], repr=False)
    detection_score: Optional[float] = Field(default=None)
    classification_score: Optional[float] = Field(default=None)
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

    def __deepcopy__(self, memo):
        obj_id = id(self)
        if obj_id in memo:
            return memo[obj_id]

        copied_data = {}
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if field_name in {"image", "cropped_image"} and isinstance(value, np.ndarray):
                copied_data[field_name] = value
            else:
                copied_data[field_name] = copy.deepcopy(value, memo)

        copied = self.__class__.model_construct(**copied_data)
        memo[obj_id] = copied
        return copied

    @field_validator("image_path", mode="before")
    @classmethod
    def parse_image_path(cls, image_path):
        if isinstance(image_path, Path) or (isinstance(image_path, str) and not is_base64(image_path)):
            image_path = Pathy.fluid(image_path)
        return image_path

    @field_validator("keypoints", mode="before")
    @classmethod
    def parse_keypoints(cls, keypoints):
        if keypoints is None:
            keypoints = []
        return np.array(keypoints).astype(int).reshape((-1, 2))

    @field_validator("mask", mode="before")
    @classmethod
    def parse_mask(cls, mask):
        if mask is None:
            return []
        if not isinstance(mask, list):
            if isinstance(mask, np.ndarray) and len(mask.shape) in [2, 3]:
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = (mask > 0).all(axis=2).astype(np.uint8)
                mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                polygons = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
                polygons = polygons[0] if len(polygons) == 2 else polygons[1]
                polygons = [np.array(polygon, dtype=np.int32) for polygon in polygons]
                return polygons
        return [np.array(points, dtype=np.int32).reshape((-1, 2)) for points in mask if len(points) > 0]

    @field_validator("image", mode="before")
    @classmethod
    def parse_image(cls, image):
        if image is not None and not isinstance(image, np.ndarray):
            image = np.array(image)
        return image

    @model_validator(mode="after")
    def set_fields(self):
        if self.image is not None:
            meta_width, meta_height = get_meta_image_size(
                image_path=self.image_path,
                image=self.image,
                meta_height=self.meta_height,
                meta_width=self.meta_width,
            )
            object.__setattr__(self, "meta_width", meta_width)
            object.__setattr__(self, "meta_height", meta_height)
        return self

    def get_image_size(self, exif_transpose: bool = False) -> Tuple[int, int]:
        """
        Returns (width, height) of image without opening it fully.
        """
        meta_width, meta_height = get_meta_image_size(
            image_path=self.image_path,
            image=self.image,
            meta_height=self.meta_height,
            meta_width=self.meta_width,
            exif_transpose=exif_transpose,
        )
        if not exif_transpose:
            self.meta_width, self.meta_height = meta_width, meta_height
        return meta_width, meta_height

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
        self, inplace: bool = False, returns_none_if_empty: bool = False, exif_transpose: bool = False
    ) -> Optional[np.ndarray]:
        if self.image is not None and isinstance(self.image, np.ndarray):
            if not inplace:
                return self.image
            else:
                image = self.image.copy()
        elif isinstance(self.image, bytes) or isinstance(self.image, str):
            image = open_image(image=self.image, open_as_rgb=True, exif_transpose=exif_transpose)
        elif self.image_path is not None:
            image = open_image(image=self.image_path, open_as_rgb=True, exif_transpose=exif_transpose)
        else:
            if returns_none_if_empty:
                return None
            raise ValueError("Object doesn't have any image.")

        if inplace:
            self.image = image

        if image is not None:
            self.get_image_size(exif_transpose=exif_transpose)

        return image

    def json(self, include_image_path: bool = True, force_include_meta: bool = False, **kwargs) -> str:
        kwargs = kwargs.copy()
        exclude = set(kwargs.pop("exclude", set()))
        if force_include_meta:
            self.get_image_size()  # write meta inplace if empty
        if not include_image_path:
            exclude.add("image_path")
        exclude_none = kwargs.pop("exclude_none", True)
        indent = kwargs.pop("indent", None)
        ensure_ascii = kwargs.pop("ensure_ascii", True)
        data = self.model_dump(exclude=exclude, exclude_none=exclude_none, mode="python", **kwargs)
        data = self._make_jsonable(data)
        return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

    @classmethod
    def _make_jsonable(cls, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (Pathy, fsspec.core.OpenFile)):
            return get_image_path_as_str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if isinstance(value, io.BytesIO):
            return value.getvalue().decode(errors="replace")
        if isinstance(value, PIL.Image.Image):
            return get_image_name(value)
        if isinstance(value, dict):
            return {str(cls._make_jsonable(key)): cls._make_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._make_jsonable(item) for item in value]
        return value

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
                with fsspec.open(d, "r") as f:
                    d = json.loads(f.read())
        elif isinstance(d, fsspec.core.OpenFile):
            with d as f:
                d = json.load(f)
        if image_path is not None:
            d["image_path"] = image_path

        return cls(**d)

    def is_empty(self):
        return self.image_path is None and self.image is None

    def _source_fields_for_propagation(self, changed_field: Optional[str] = None) -> Dict[str, Any]:
        if changed_field == "image":
            fields = ("image", "meta_width", "meta_height")
        elif changed_field in self._SOURCE_FIELDS:
            fields = (changed_field,)
        else:
            fields = self._SOURCE_FIELDS
        return {field: getattr(self, field) for field in fields}

    def open_mask(
        self,
        exif_transpose: bool = False,
    ) -> np.ndarray:
        if isinstance(self.mask, np.ndarray):
            return self.mask
        elif not isinstance(self.mask, list):
            mask = open_image(image=self.mask, open_as_rgb=True, exif_transpose=exif_transpose)
            return mask
        else:
            width, height = self.get_image_size(exif_transpose=exif_transpose)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = cv2.fillPoly(mask, self.mask, 255)
            return mask


class BboxData(BaseImageData):
    xmin: Optional[Union[int, float]] = None
    ymin: Optional[Union[int, float]] = None
    xmax: Optional[Union[int, float]] = None
    ymax: Optional[Union[int, float]] = None
    cropped_image: Optional[np.ndarray] = Field(default=None, repr=False)
    additional_bboxes_data: List["BboxData"] = Field(default_factory=list)

    @field_validator("xmin", "ymin", mode="before")
    @classmethod
    def parse_xmin(cls, v):
        if v is not None:
            v = max(0, v)
        return v

    @model_validator(mode="after")
    def set_additional_bboxes_data_source_fields(self):
        self._propagate_source_fields_to_additional_bboxes()
        return self

    def _apply_source_fields(self, source_fields: Dict[str, Any]) -> None:
        for name, value in source_fields.items():
            object.__setattr__(self, name, value)
        self._propagate_source_fields_to_additional_bboxes(source_fields)

    def _propagate_source_fields_to_additional_bboxes(self, source_fields: Optional[Dict[str, Any]] = None) -> None:
        if not hasattr(self, "additional_bboxes_data"):
            return
        source_fields = source_fields or self._source_fields_for_propagation()
        for additional_bbox_data in self.additional_bboxes_data:
            additional_bbox_data._apply_source_fields(source_fields)

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
        source_image: np.ndarray = None,
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
        xmin_in_cropped_image = max(0, min(xmin_offset, self.xmin - xmin_offset))
        ymin_in_cropped_image = max(0, min(ymin_offset, self.ymin - ymin_offset))
        xmax_in_cropped_image = max(0, min(xmax_offset, width - self.xmax))
        ymax_in_cropped_image = max(0, min(ymax_offset, height - self.ymax))
        return (
            max(0, round(self.xmin - xmin_in_cropped_image)),
            max(0, round(self.ymin - ymin_in_cropped_image)),
            min(width - 1, round(self.xmax + xmax_in_cropped_image)),
            min(height - 1, round(self.ymax + ymax_in_cropped_image)),
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
        image_data_cls: Type["ImageData"] = "ImageData",
    ) -> Optional[Union[np.ndarray, "BboxData"]]:
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
                keypoints = copy.deepcopy(self.keypoints)
                keypoints[:, 0] -= xmin
                keypoints[:, 1] -= ymin
                mask = copy.deepcopy(self.mask)
                for polygon in mask:
                    polygon[:, 0] -= xmin
                    polygon[:, 1] -= ymin

                copy_image = self.image  # Memory trick
                copy_cropped_image = self.cropped_image  # Memory trick
                self.image = None  # Memory trick
                self.cropped_image = None  # Memory trick
                additional_bboxes_data = copy.deepcopy(self.additional_bboxes_data)

                def crop_additional_bbox_data(bbox_data: BboxData):
                    bbox_data.image_path = None
                    bbox_data.image = cropped_image
                    bbox_data.xmin -= xmin
                    bbox_data.ymin -= ymin
                    bbox_data.xmax -= xmin
                    bbox_data.ymax -= ymin
                    bbox_data.keypoints[:, 0] -= xmin
                    bbox_data.keypoints[:, 1] -= ymin
                    if isinstance(bbox_data.mask, list):
                        for polygon in bbox_data.mask:
                            polygon[:, 0] -= xmin
                            polygon[:, 1] -= ymin
                    for additional_bbox_data in bbox_data.additional_bboxes_data:
                        crop_additional_bbox_data(additional_bbox_data)

                for additional_bbox_data in additional_bboxes_data:
                    crop_additional_bbox_data(additional_bbox_data)

                if image_data_cls == "ImageData":
                    from cv_pipeliner.core.data import ImageData

                    image_data_cls = ImageData

                image_data = image_data_cls(
                    image=cropped_image,
                    bboxes_data=additional_bboxes_data,
                    label=self.label,
                    keypoints=keypoints,
                    mask=mask,
                    additional_info=self.additional_info,
                )
                self.image = copy_image  # Memory trick
                self.cropped_image = copy_cropped_image  # Memory trick
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

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in ["image_path", "image", "meta_width", "meta_height"]:
            self._propagate_source_fields_to_additional_bboxes(self._source_fields_for_propagation(name))
        elif name == "additional_bboxes_data":
            self._propagate_source_fields_to_additional_bboxes()

    def open_cropped_mask(
        self,
        xmin_offset: Union[int, float] = 0,
        ymin_offset: Union[int, float] = 0,
        xmax_offset: Union[int, float] = 0,
        ymax_offset: Union[int, float] = 0,
        exif_transpose: bool = False,
        source_image: np.ndarray = None,
    ) -> np.ndarray:
        xmin, ymin, xmax, ymax = self.coords_with_offset(
            xmin_offset, ymin_offset, xmax_offset, ymax_offset, source_image
        )
        if self.mask is None or isinstance(self.mask, np.ndarray):
            return self.mask
        elif not isinstance(self.mask, list):
            mask = open_image(image=self.mask, open_as_rgb=True, exif_transpose=exif_transpose)
            mask_height, mask_width = mask.shape[0:2]
            if (mask_width, mask_height) == self.get_image_size(exif_transpos=exif_transpose):
                mask = mask[ymin:ymax, xmin:xmax]
            return mask
        else:
            width, height = (xmax - xmin), (ymax - ymin)
            polygons = copy.deepcopy(self.mask)
            for polygon in polygons:
                polygon[:, 0] = np.clip(polygon[:, 0] - xmin, 0, width - 1)
                polygon[:, 1] = np.clip(polygon[:, 1] - ymin, 0, height - 1)
            mask = np.zeros((height, width), dtype=np.uint8)
            mask = cv2.fillPoly(mask, polygons, 255)
            return mask


class ImageData(BaseImageData):
    bboxes_data: List[BboxData] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_bboxes_data_source_fields(self):
        self._propagate_source_fields_to_bboxes()
        return self

    def _propagate_source_fields_to_bboxes(self, source_fields: Optional[Dict[str, Any]] = None) -> None:
        if not hasattr(self, "bboxes_data"):
            return
        source_fields = source_fields or self._source_fields_for_propagation()
        for bbox_data in self.bboxes_data:
            bbox_data._apply_source_fields(source_fields)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in ["image_path", "image", "meta_width", "meta_height"]:
            self._propagate_source_fields_to_bboxes(self._source_fields_for_propagation(name))
        elif name == "bboxes_data":
            self._propagate_source_fields_to_bboxes()

    def find_bbox_data_by_coords(self, xmin: int, ymin: int, xmax: int, ymax: int) -> BboxData:
        bboxes_data_coords = [bbox_data.coords for bbox_data in self.bboxes_data]
        return self.bboxes_data[bboxes_data_coords.index((xmin, ymin, xmax, ymax))]

    def count_children_bboxes_data(self) -> int:
        count = 0
        for bbox_data in self.bboxes_data:
            count += 1 + bbox_data.count_children()
        return count

    def open_mask(
        self,
        exif_transpose: bool = False,
        include_bboxes_data: bool = True,
        include_additional_bboxes_data: bool = True,
        additional_bboxes_data_depth: Optional[int] = None,
    ) -> np.ndarray:
        mask = BaseImageData.open_mask(self, exif_transpose=exif_transpose)
        if include_additional_bboxes_data:
            from cv_pipeliner.utils.images_datas import (
                flatten_additional_bboxes_data_in_image_data,
            )

            bboxes_data = flatten_additional_bboxes_data_in_image_data(
                self, additional_bboxes_data_depth=additional_bboxes_data_depth
            ).bboxes_data
        else:
            bboxes_data = self.bboxes_data
        if include_bboxes_data:
            for bbox_data in bboxes_data:
                new_mask = bbox_data.open_mask(exif_transpose=exif_transpose)
                mask[new_mask > 0] = new_mask[new_mask > 0]
        return mask
