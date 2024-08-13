import json
from pathlib import Path
from typing import Dict, Type, Union

import fsspec

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.core.data_converter import DataConverter


class JSONDataConverter(DataConverter):
    def __init__(self, image_data_cls: Type[ImageData] = ImageData):
        self.image_data_cls = image_data_cls

    def get_annot_from_image_data(self, image_data: ImageData) -> Dict:
        image_data = self.filter_image_data(image_data)
        return json.loads(image_data.json())

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self, image_path: Union[str, Path], annot: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> ImageData:
        return self.image_data_cls.from_json(annot, image_path=image_path)
