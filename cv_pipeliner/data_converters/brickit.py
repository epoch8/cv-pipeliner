import json

from typing import Union, Dict, List
from pathlib import Path
from pathy import Pathy

import fsspec

from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.core.data import BboxData, ImageData


class BrickitDataConverter(DataConverter):
    def __init__(self,
                 class_names: List[str] = None,
                 class_mapper: Dict[str, str] = None,
                 skip_nonexists: bool = False):
        super().__init__(
            class_names=class_names,
            class_mapper=class_mapper,
            skip_nonexists=skip_nonexists
        )

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path, fsspec.core.OpenFile],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> ImageData:
        if isinstance(image_path, fsspec.core.OpenFile):
            image_name = Pathy(image_path.path).name
        else:
            image_name = Pathy(image_path).name

        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, 'r', encoding='utf8') as f:
                annot = json.load(f)
        if isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                annot = json.load(f)
        image_idx = None
        for i, image_annot in enumerate(annot):
            if image_annot['filename'] == image_name:
                image_idx = i
                break
        if image_idx is None:
            return None

        annot = annot[image_idx]
        additional_info = {}
        for key in annot:
            if key not in ['objects', 'filename']:
                additional_info[key] = annot[key]
        bboxes_data = []
        for obj in annot['objects']:
            if 'bbox' in obj:
                xmin, ymin, xmax, ymax = obj['bbox']
            else:
                xmin, ymin, xmax, ymax = obj
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            label = obj['label'] if 'label' in obj else None
            angle = obj['angle'] if 'angle' in obj else 0
            labels_top_n = obj['labels_top_n'] if 'labels_top_n' in obj else None
            top_n = len(labels_top_n) if labels_top_n is not None else None
            additional_info = {}
            if isinstance(obj, dict):
                for key in obj:
                    if key not in ['bbox', 'label', 'angle', 'labels_top_n', 'top_n']:
                        additional_info[key] = obj[key]
            bboxes_data.append(BboxData(
                image_path=image_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                angle=angle,
                label=label,
                labels_top_n=labels_top_n,
                top_n=top_n,
                additional_info=additional_info
            ))

        image_data = ImageData(
            image_path=image_path,
            bboxes_data=bboxes_data,
            additional_info=additional_info
        )

        return image_data

    def get_annot_from_image_data(
        self,
        image_data: ImageData
    ) -> Dict:
        annot = {
            'filename': image_data.image_name,
            'objects': [{
                'bbox': [int(bbox_data.xmin), int(bbox_data.ymin), int(bbox_data.xmax), int(bbox_data.ymax)],
                'label': str(bbox_data.label),
                'angle': int(bbox_data.angle),
                'labels_top_n': list(bbox_data.labels_top_n) if bbox_data.labels_top_n is not None else None,
                **{
                    key: bbox_data.additional_info[key] for key in bbox_data.additional_info
                }
            } for bbox_data in image_data.bboxes_data]
        }
        if len(image_data.additional_info) > 0:
            for key in image_data.additional_info:
                annot[key] = image_data.additional_info[key]
        return annot
