import json
import math
import numpy as np

from typing import Dict, Union, Tuple, List
from pathlib import Path

from two_stage_pipeliner.core.data import ImageData, BboxData


def json_results_to_images_data(
    result_json: Union[Dict, str, Path],
    images_source_dir: Union[str, Path]
) -> List[ImageData]:
    if isinstance(result_json, str) or isinstance(result_json, Path):
        with open(result_json, 'r') as src:
            result_json = json.load(src)
    images_source_dir = Path(images_source_dir)
    images_data = []

    def rotate_point(
        x: float, y: float, cx: float, cy: float, angle: float
    ) -> Tuple[float, float]:
        angle = math.radians(angle)
        xnew = cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle)
        ynew = cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
        return xnew, ynew

    for ann_item in result_json:
        filename = Path(ann_item['data']['image']).name
        image_path = images_source_dir / filename
        completions = ann_item['completions']

        assert len(completions) == 1

        completion = completions[0]

        if 'skipped' in completion and completion['skipped']:
            continue

        results = completion['result']
        bboxes_data = []
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
        images_data.append(ImageData(
            image_path=image_path,
            bboxes_data=bboxes_data
        ))

    return images_data
