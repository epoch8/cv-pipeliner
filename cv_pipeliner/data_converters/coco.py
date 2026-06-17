import json
from pathlib import Path
from typing import Dict, List, Union

import fsspec
from joblib import Parallel, delayed
from tqdm import tqdm

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.core.data_converter import DataConverter


class COCODataConverter(DataConverter):
    @staticmethod
    def _load_annot(annot: Union[Path, str, Dict, fsspec.core.OpenFile]) -> Dict:
        if isinstance(annot, str) or isinstance(annot, Path):
            with fsspec.open(annot, "r", encoding="utf8") as f:
                return json.load(f)
        if isinstance(annot, fsspec.core.OpenFile):
            with annot as f:
                return json.load(f)
        return annot

    @staticmethod
    def _polygon_to_points(polygon: List[float]) -> List[List[int]]:
        return [[round(polygon[i]), round(polygon[i + 1])] for i in range(0, len(polygon), 2)]

    @DataConverter.assert_image_data
    def get_image_data_from_annot(
        self,
        image_path: Union[str, Path],
        annot: Union[Path, str, Dict, fsspec.core.OpenFile],
    ) -> ImageData:
        image_path = Path(image_path)
        annot = self._load_annot(annot)
        category_by_id = {category["id"]: category["name"] for category in annot["categories"]}
        image_id = int(image_path.stem)
        bboxes_data = []
        for bbox_annot in annot["annotations"]:
            if bbox_annot["image_id"] != image_id:
                continue
            if bbox_annot.get("iscrowd", 0) != 0 or not isinstance(bbox_annot.get("segmentation"), list):
                continue

            xmin, ymin, width, height = bbox_annot["bbox"]
            bboxes_data.append(
                BboxData(
                    image_path=image_path,
                    xmin=round(xmin),
                    ymin=round(ymin),
                    xmax=round(xmin + width),
                    ymax=round(ymin + height),
                    label=category_by_id[bbox_annot["category_id"]],
                    mask=[
                        self._polygon_to_points(polygon)
                        for polygon in bbox_annot["segmentation"]
                        if len(polygon) >= 6
                    ],
                    additional_info={"coco_annotation_id": bbox_annot["id"]},
                )
            )

        return ImageData(
            image_path=image_path,
            label=bboxes_data[0].label if len(bboxes_data) > 0 else None,
            additional_info={"coco_image_id": image_id},
            bboxes_data=bboxes_data,
        )

    @DataConverter.assert_images_data
    def get_images_data_from_annots(
        self,
        image_paths: List[Union[str, Path]],
        annots: Union[Path, str, Dict, fsspec.core.OpenFile],
        n_jobs: int = 8,
        disable_tqdm: bool = False,
    ) -> List[ImageData]:
        annot = self._load_annot(annots)
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.get_image_data_from_annot)(image_path=image_path, annot=annot)
            for image_path in tqdm(image_paths, total=len(image_paths), disable=disable_tqdm)
        )
