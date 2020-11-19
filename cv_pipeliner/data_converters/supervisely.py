import json

from typing import Union, Dict, List
from pathlib import Path

import fsspec
from pathy import Pathy

from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.core.data import BboxData, ImageData


class SuperviselyDataConverter(DataConverter):
    def __init__(self,
                 class_names: List[str] = None,
                 class_mapper: Dict[str, str] = None,
                 default_value: str = "",
                 skip_nonexists: bool = False):
        super().__init__(
            class_names=class_names,
            class_mapper=class_mapper,
            default_value=default_value,
            skip_nonexists=skip_nonexists
        )

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path, fsspec.core.OpenFile],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, 'r', encoding='utf8') as f:
                annot = json.load(f)
        if isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                annot = json.load(f)

        bboxes_data = []
        for obj in annot['objects']:
            (xmin, ymin), (xmax, ymax) = obj['points']['exterior']
            label = obj['tags'][0]['name'] if obj['tags'] else None
            bboxes_data.append(BboxData(
                image_path=image_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                label=label
            ))

        image_data = ImageData(
            image_path=image_path,
            bboxes_data=bboxes_data
        )

        return image_data
