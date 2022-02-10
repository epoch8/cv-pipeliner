import json

from typing import Union, Dict, List
from pathlib import Path
from pathy import Pathy

import fsspec

from cv_pipeliner.core.data_converter import DataConverter
from cv_pipeliner.core.data import BboxData, ImageData


class BrickitDataConverter(DataConverter):
    def get_annot_from_image_data(
        self,
        image_data: ImageData
    ) -> Dict:
        annot = {
            'filename': image_data.image_name,
            'objects': []
        }
        image_data = self.filter_image_data(image_data)
        for bbox_data in image_data.bboxes_data:
            bbox_data_json = bbox_data.json()
            obj = {
                'bbox': [
                    bbox_data_json['xmin'], bbox_data_json['ymin'], bbox_data_json['xmax'], bbox_data_json['ymax']
                ],
                'label': bbox_data_json['label'],
                **({
                    key: bbox_data_json['additional_info'][key]
                    for key in bbox_data_json['additional_info']
                } if 'additional_info' in bbox_data_json else {})
            }
            if bbox_data.labels_top_n is not None:
                obj['labels_top_n'] = bbox_data_json['labels_top_n']

            annot['objects'].append(obj)
        if len(image_data.additional_info) > 0:
            for key in image_data.additional_info:
                annot[key] = image_data.additional_info[key]
        return annot

    @DataConverter.assert_images_data
    def get_images_data_from_annots(
        self,
        images_dir: Union[str, Path, fsspec.core.OpenFile],
        annots: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> List[ImageData]:
        if isinstance(annots, str) or isinstance(annots, Path):
            with fsspec.open(annots, 'r', encoding='utf8') as f:
                annots = json.load(f)
        elif isinstance(annots, fsspec.core.OpenFile):
            with annots as f:
                annots = json.load(f)
        images_dir = Pathy(images_dir)

        images_data = []
        for annot in annots:
            image_name = annot['filename']
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
                labels_top_n = obj['labels_top_n'] if 'labels_top_n' in obj else None
                top_n = len(labels_top_n) if labels_top_n is not None else None
                keypoints = obj['keypoints'] if 'keypoints' in obj else []
                bbox_additional_info = {}
                if isinstance(obj, dict):
                    for key in obj:
                        if key not in ['bbox', 'label', 'labels_top_n', 'top_n']:
                            bbox_additional_info[key] = obj[key]
                bboxes_data.append(BboxData(
                    image_path=images_dir / image_name,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    keypoints=keypoints,
                    label=label,
                    labels_top_n=labels_top_n,
                    top_n=top_n,
                    additional_info=bbox_additional_info
                ))

            images_data.append(ImageData(
                image_path=images_dir / image_name,
                bboxes_data=bboxes_data,
                additional_info=additional_info
            ))

        return images_data

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile]
    ) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, 'r', encoding='utf8') as f:
                annots = json.load(f)
        elif isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                annots = json.load(f)
        elif isinstance(annot, dict):
            annots = [annot]
        else:
            annots = annot
        image_path = Pathy(image_path)
        annots = [annot for annot in annots if annot['filename'] == image_path.name]

        return self.get_images_data_from_annots(
            images_dir=image_path.parent,
            annots=annots
        )[0]
