import json

from typing import Union, Dict, List
from pathlib import Path

from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.core.data import BboxData, ImageData


class BrickitDataConverter(DataConverter):
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
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict]
    ) -> ImageData:
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
            return None

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
            additional_info = {}
            if isinstance(obj, dict):
                for key in obj:
                    if 'bbox' != key and 'label' != key:
                        additional_info[key] = obj[key]
            image_data.bboxes_data.append(BboxData(
                image_path=image_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                label=label,
                additional_info=additional_info
            ))

        return image_data

    def get_annot_from_image_data(
        self,
        image_data: ImageData
    ) -> Dict:
        annot = {
            'filename': image_data.image_path.name,
            'objects': [{
                'bbox': [int(bbox_data.xmin), int(bbox_data.ymin), int(bbox_data.xmax), int(bbox_data.ymax)],
                'label': bbox_data.label,
                **{
                    key: bbox_data.additional_info[key] for key in bbox_data.additional_info
                }
            } for bbox_data in image_data.bboxes_data]
        }
        return annot
