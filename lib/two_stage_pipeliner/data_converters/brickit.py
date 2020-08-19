import json

from typing import Union, Dict
from pathlib import Path

from two_stage_pipeliner.core.data_converter import DataConverter
from two_stage_pipeliner.core.data import BboxData, ImageData


class BrickitDataConverter(DataConverter):
    def __init__(self):
        DataConverter.__init__(self)

    @DataConverter.assert_image_data
    def get_image_data(self,
                       image_path: Union[Path, str],
                       annot: Union[Path, str, Dict]) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with open(annot, 'r', encoding='utf8') as f:
                annot = json.load(f)

        image_path = Path(image_path)
        image_idx = None
        for i, image_annot in enumerate(annot):
            if image_annot['filename'] == image_path.name:
                image_idx = i
                break
        if image_idx is None:
            raise ValueError(
                f"Image {image_path} does not have annotation in given annot."
            )

        annot = annot[image_idx]

        image_data = ImageData(
            image_path=image_path,
            image=None,
            bboxes_data=[]
        )
        for obj in annot['objects']:
            if 'bbox' in obj:
                xmin, ymin, xmax, ymax = obj['bbox']
            else:
                xmin, ymin, xmax, ymax = obj
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            if 'label' in obj:
                label = obj['label']
            else:
                label = 'brick'
            image_data.bboxes_data.append(BboxData(
                image_path=image_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                label=label
            ))

        return image_data
