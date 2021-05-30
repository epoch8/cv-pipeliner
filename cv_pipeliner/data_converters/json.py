import json

from typing import Union, Dict, List
from pathlib import Path
from pathy import Pathy

import fsspec

from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.core.data import ImageData


class JSONDataConverter(DataConverter):
    def __init__(self,
                 class_names: List[str] = None,
                 class_mapper: Dict[str, str] = None,
                 skip_nonexists: bool = False):
        super().__init__(
            class_names=class_names,
            class_mapper=class_mapper,
            skip_nonexists=skip_nonexists
        )

    def get_annot_from_image_data(
        self,
        image_data: ImageData
    ) -> Dict:
        return image_data.json()

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, 'r', encoding='utf8') as f:
                annots = json.load(f)
        if isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                annots = json.load(f)
        image_path = Pathy(image_path)
        return ImageData.from_json(annots, image_path=image_path)
