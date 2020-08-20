
import abc

from typing import Union, List, Dict, Any
from pathlib import Path

from two_stage_pipeliner.logging import logger
from two_stage_pipeliner.core.data import BboxData, ImageData


def assert_image_data(fn):
    def wrapped(
        data_converter: "DataConverter",
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict]
    ) -> ImageData:
        image_data = fn(data_converter, image_path, annot)
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

            if xmin >= xmax or ymin >= ymax:
                logger.warning(
                    f"Wrong annotation: "
                    f"incorrect bbox (xmin, ymin, xmax, ymax): {(xmin, ymin, xmax, ymax)} "
                    "(xmin >= xmax or ymin >= ymax). Skipping."
                )
                continue

            if data_converter.class_names and data_converter.class_mapper:
                bbox_data.label = data_converter._filter_label_by_class_mapper(
                    bbox_data.label,
                    data_converter.class_names,
                    data_converter.class_mapper,
                    data_converter.default_value
                )

            if (
                data_converter.class_names and
                bbox_data.label not in data_converter.class_names and
                data_converter.skip_nonexists
            ):
                continue

            new_bboxes_data.append(bbox_data)

        image_data = ImageData(
            image_path=image_data.image_path,
            bboxes_data=new_bboxes_data
        )

        return image_data

    return wrapped


class DataConverter(abc.ABC):
    def __init__(self,
                 class_names: List[str] = None,
                 class_mapper: Dict[str, str] = None,
                 default_value: str = "",
                 skip_nonexists: bool = False):
        self.class_names = class_names
        self.class_mapper = class_mapper
        self.default_value = default_value
        self.skip_nonexists = skip_nonexists

    def _filter_label_by_class_mapper(
        self,
        label: str,
        class_names: List[str],
        class_mapper: Dict[str, str] = None,
        default_value: str = ""
    ) -> str:
        if label in class_names:
            return label
        if class_mapper is None:
            return default_value
        else:
            return class_mapper.get(label, default_value)

    @abc.abstractmethod
    @assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[Path, str],
        annot: Union[Path, str, Dict]
    ) -> ImageData:
        pass

    def get_images_data_from_annots(
        self,
        image_paths: List[Union[Path, str]],
        annots: List[Union[Path, str, Dict]]
    ) -> List[ImageData]:
        assert len(image_paths) == len(annots)

        images_data = [
            self.get_image_data_from_annot(image_path, annot)
            for image_path, annot in zip(image_paths, annots)
        ]
        return images_data

    def get_n_bboxes_data_from_annots(
        self,
        image_paths: List[Union[Path, str]],
        annots: List[Union[Path, str, Dict]]
    ) -> List[List[BboxData]]:
        assert len(image_paths) == len(annots)

        images_data = self.get_images_data_from_annots(image_paths, annots)
        n_bboxes_data = [image_data.bboxes_data for image_data in images_data]
        return n_bboxes_data
