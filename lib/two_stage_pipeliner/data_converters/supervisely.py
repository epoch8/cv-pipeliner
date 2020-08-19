import json

from typing import Union, Dict
from pathlib import Path

from two_stage_pipeliner.core.data_converter import DataConverter
from two_stage_pipeliner.core.data import BboxData, ImageData


class SuperviselyDataConverter(DataConverter):
    def __init__(self):
        DataConverter.__init__(self)

    @DataConverter.assert_image_data
    def get_image_data(self,
                       image_path: Union[Path, str],
                       annot: Union[Path, str, Dict]) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with open(annot, 'r', encoding='utf8') as f:
                annot = json.load(f)
        image_data = ImageData(
            image_path=image_path,
            image=None,
            bboxes_data=[]
        )
        for obj in annot['objects']:
            (xmin, ymin), (xmax, ymax) = obj['points']['exterior']
            label = obj['tags'][0]['name'] if obj['tags'] else None
            image_data.bboxes_data.append(BboxData(
                image_path=image_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                label=label
            ))

        return image_data
