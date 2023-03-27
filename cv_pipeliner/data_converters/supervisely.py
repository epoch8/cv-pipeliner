import json

from typing import Union, Dict, List
from pathlib import Path

import fsspec

from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.core.data import BboxData, ImageData


class SuperviselyDataConverter(DataConverter):
    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self, image_path: Union[str, Path, fsspec.core.OpenFile], annot: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, "r", encoding="utf8") as f:
                annot = json.load(f)
        if isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                annot = json.load(f)

        bboxes_data = []
        for obj in annot["objects"]:
            (xmin, ymin), (xmax, ymax) = obj["points"]["exterior"]
            label = obj["tags"][0]["name"] if obj["tags"] else None
            bboxes_data.append(BboxData(image_path=image_path, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label))

        image_data = ImageData(image_path=image_path, bboxes_data=bboxes_data)

        return image_data

    def get_annot_from_image_data(self, image_data: ImageData) -> Dict:
        image_data = self.filter_image_data(image_data)
        image = image_data.open_image()
        height, width, _ = image.shape
        annot = {
            "description": "",
            "tags": [],
            "size": {"height": height, "width": width},
            "objects": [
                {
                    "description": "",
                    "geometryType": "rectangle",
                    "tags": [
                        {
                            "name": str(bbox_data.label),
                            "value": None,
                        }
                    ],
                    "classTitle": "bbox",
                    "points": {
                        "exterior": [
                            [int(bbox_data.xmin), int(bbox_data.ymin)],
                            [int(bbox_data.xmax), int(bbox_data.ymax)],
                        ],
                        "interior": [],
                    },
                }
                for bbox_data in image_data.bboxes_data
            ],
        }
        return annot
