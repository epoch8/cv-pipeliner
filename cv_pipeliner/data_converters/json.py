import json

from typing import Union, Dict
from pathlib import Path
from pathy import Pathy

import fsspec

from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.core.data import ImageData


class JSONDataConverter(DataConverter):
    def __init__(self):
        pass

    def get_annot_from_image_data(
        self,
        image_data: ImageData
    ) -> Dict:
        image_data = self.filter_image_data(image_data)
        return image_data.json()

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> ImageData:
        return ImageData.from_json(annot, image_path=image_path)
