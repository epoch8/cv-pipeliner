import json
import numpy as np

from typing import Union, Dict, List
from pathlib import Path

from two_stage_pipeliner.core.data_converter import DataConverter, assert_image_data
from two_stage_pipeliner.core.data import BboxData, ImageData
from two_stage_pipeliner.utils.images import rotate_point


class LabelStudioDetectionDataConverter(DataConverter):
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

    @assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[Path, str],
        annot: Union[Path, str, Dict],
    ) -> ImageData:
        if isinstance(annot, str) or isinstance(annot, Path):
            with open(annot, 'r') as src:
                result_json = json.load(src)

        image_path = Path(image_path)
        completions = None
        for ann_item in result_json:
            filename = Path(ann_item['data']['image']).name
            if filename == image_path.name:
                completions = ann_item['completions']
        if completions is None:
            raise ValueError(
                f"Image {image_path} does not have annotation in given annot."
            )

        assert len(completions) == 1

        completion = completions[0]

        bboxes_data = []
        if 'skipped' in completion and completion['skipped']:
            results = []
        else:
            results = completion['result']

        for result in results:
            if result['from_name'] != 'label':
                continue
            original_height = result['original_height']
            original_width = result['original_width']
            height = result['value']['height']
            width = result['value']['width']
            xmin = result['value']['x']
            ymin = result['value']['y']
            angle = result['value']['rotation']
            label = result['value']['rectanglelabels'][0]

            xmax = xmin + width
            ymax = ymin + height

            xmin = xmin / 100 * original_width
            ymin = ymin / 100 * original_height
            xmax = xmax / 100 * original_width
            ymax = ymax / 100 * original_height
            points = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
            new_points = [rotate_point(x=x, y=y, cx=xmin, cy=ymin, angle=angle) for (x, y) in points]
            xmin = min([x for (x, y) in new_points])
            ymin = min([y for (x, y) in new_points])
            xmax = max([x for (x, y) in new_points])
            ymax = max([y for (x, y) in new_points])
            bbox = np.array([xmin, ymin, xmax, ymax])
            bbox = bbox.round().astype(int)
            xmin, ymin, xmax, ymax = bbox
            bboxes_data.append(BboxData(
                image_path=image_path,
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                label=label
            ))
        image_data = ImageData(
            image_path=image_path,
            bboxes_data=bboxes_data
        )

        return image_data

    def get_images_data_from_annots(
        self,
        image_paths: List[Union[Path, str]],
        annots: Union[Path, str, Dict]
    ) -> List[ImageData]:
        if isinstance(annots, str) or isinstance(annots, Path):
            with open(annots, 'r', encoding='utf8') as f:
                annots = json.load(f)

        images_data = [self.get_image_data_from_annot(image_path, annots) for image_path in image_paths]
        return images_data

    def get_n_bboxes_data_from_annots(
        self,
        image_paths: List[Union[Path, str]],
        annots: Union[Path, str, Dict]
    ) -> List[List[BboxData]]:

        images_data = self.get_images_data_from_annots(image_paths, annots)
        n_bboxes_data = [image_data.bboxes_data for image_data in images_data]
        return n_bboxes_data
