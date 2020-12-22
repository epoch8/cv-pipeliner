import json
from typing import Any, Dict, List

import fsspec

from kedro.io import AbstractDataSet

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.utils.streamlit.data import get_images_data_from_dir
from cv_pipeliner.data_converters.brickit import BrickitDataConverter


class ImageDataSet(AbstractDataSet):
    def __init__(self, annotation_type: str, images_dir: str, annotation_filepath: str):
        self.annotation_type = annotation_type
        self.images_dir = images_dir
        self.annotation_filepath = annotation_filepath

    def _load(self) -> List[ImageData]:
        self.images_data, annotation_success = get_images_data_from_dir(
            images_annotation_type=self.annotation_type,
            images_dir=self.images_dir,
            annotation_filepath=self.annotation_filepath
        )
        if not annotation_success:
            raise ValueError("Dataset doesn't loaded properly")
        return self.images_data

    def _save(self, images_data: List[ImageData]) -> None:
        annotation = BrickitDataConverter().get_annot_from_images_data(images_data)
        with fsspec.open(self.annotation_filepath, 'w') as out:
            json.dump(annotation, out)

    def _describe(self) -> Dict[str, Any]:
        dict(
            annotation_type=self.annotation_type,
            annotation_filepath=self.annotation_filepath,
            images_dir=self.images_dir
        )
