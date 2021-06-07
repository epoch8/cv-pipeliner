
import abc
import json

from typing import Union, List, Dict
from pathlib import Path

import fsspec

from cv_pipeliner.logging import logger
from cv_pipeliner.core.data import BboxData, ImageData


class DataConverter(abc.ABC):
    def __init__(
        self,
        class_names: List[str] = None,
        class_mapper: Dict[str, str] = None,
        skip_nonexists: bool = False
    ):
        self.class_names = class_names
        self.class_mapper = class_mapper
        self.skip_nonexists = skip_nonexists

    def filter_image_data(
        self,
        image_data: ImageData
    ) -> ImageData:
        if image_data is None:
            return None
        looked_bboxes = set()
        new_bboxes_data = []
        for bbox_data in image_data.bboxes_data:
            xmin, ymin, xmax, ymax = bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            if (xmin, ymin, xmax, ymax) in looked_bboxes:
                logger.warning(
                    f'Repeated bbox detected at image {bbox_data.image_path}: '
                    f'(xmin, ymin, xmax, ymax) = {(xmin, xmin, ymax, xmax)}. Skipping.'
                )
                continue
            else:
                looked_bboxes.add((xmin, ymin, xmax, ymax))

            if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0:
                logger.warning(
                    f"Wrong annotation at image {bbox_data.image_path}: "
                    f"incorrect bbox (xmin, ymin, xmax, ymax): {(xmin, ymin, xmax, ymax)} "
                    "(xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0). Skipping."
                )
                continue

            if self.class_mapper is not None:
                if bbox_data.label in self.class_mapper:
                    bbox_data.label = self.class_mapper[bbox_data.label]
            if (
                self.class_names is not None and
                (
                    bbox_data.label not in self.class_names and self.skip_nonexists
                )
            ):
                continue

            new_bboxes_data.append(bbox_data)

        image_data = ImageData(
            image_path=image_data.image_path,
            bboxes_data=new_bboxes_data,
            additional_info=image_data.additional_info
        )

        return image_data

    def assert_image_data(fn):
        def wrapped(
            data_converter: "DataConverter",
            *args,
            **kwargs
        ) -> ImageData:
            image_data = fn(data_converter, *args, **kwargs)
            image_data = data_converter.filter_image_data(image_data)
            return image_data

        return wrapped

    def assert_images_data(fn):
        def wrapped(
            data_converter: "DataConverter",
            *args,
            **kwargs
        ) -> List[ImageData]:
            images_data = fn(data_converter, *args, **kwargs)
            images_data = [
                data_converter.filter_image_data(image_data) for image_data in images_data
            ]
            return images_data

        return wrapped

    @abc.abstractmethod
    @assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> ImageData:
        pass

    def get_images_data_from_annots(
        self,
        image_paths: List[Union[str, Path]],
        annots: Union[List[Union[Path, str, Dict]], Union[Path, str, Dict]]
    ) -> List[ImageData]:
        if isinstance(annots, str) or isinstance(annots, Path):
            with fsspec.open(annots, 'r', encoding='utf8') as f:
                annots = json.load(f)
        if isinstance(annots, List):
            assert len(image_paths) == len(annots)

        images_data = [
            self.get_image_data_from_annot(
                image_path=image_path,
                annot=annot
            )
            for image_path, annot in zip(image_paths, annots)
        ]
        images_data = [image_data for image_data in images_data if image_data is not None]
        return images_data

    def get_n_bboxes_data_from_annots(
        self,
        image_paths: List[Union[str, Path]],
        annots: Union[List[Union[Path, str, Dict]], Union[Path, str, Dict]]
    ) -> List[List[BboxData]]:
        images_data = self.get_images_data_from_annots(
            image_paths=image_paths,
            annots=annots
        )
        n_bboxes_data = [image_data.bboxes_data for image_data in images_data]
        return n_bboxes_data

    def get_annot_from_image_data(
        self,
        image_data: ImageData
    ) -> Dict:
        return {}

    def get_annot_from_images_data(
        self,
        images_data: ImageData
    ) -> Dict:
        annots = [self.get_annot_from_image_data(image_data) for image_data in images_data]
        return annots

    def get_annot_from_n_bboxes_data(
        self,
        image_paths: List[Union[str, Path]],
        n_bboxes_data: List[List[BboxData]],
    ) -> List[List[BboxData]]:
        images_data = [
            ImageData(image_path=image_path, bboxes_data=bboxes_data)
            for image_path, bboxes_data in zip(image_paths, n_bboxes_data)
        ]
        return self.get_annot_from_images_data(images_data)
