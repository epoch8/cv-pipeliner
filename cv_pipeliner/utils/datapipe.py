import json
import pathlib
import numpy as np
import fsspec
from typing import Any, Dict, IO, Tuple, List

from datapipe.store.filedir import ItemStoreFileAdapter
from cv_pipeliner.core.data import ImageData
from cv_pipeliner.utils.imagesize import get_image_size


class ImageDataFile(ItemStoreFileAdapter):
    '''
    Converts each ImageData file into Pandas record
    '''

    mode = 't'

    def load(self, f: IO) -> ImageData:
        image_data_json = json.load(f)
        image_data = ImageData.from_json(image_data_json) if image_data_json is not None else None
        return {'image_data': image_data}

    def dump(self, obj: Dict[str, Any], f: IO) -> None:
        image_data: ImageData = obj['image_data']
        return json.dump(image_data.json() if image_data is not None else None, f, indent=4, ensure_ascii=False)


class NumpyDataFile(ItemStoreFileAdapter):
    '''
    Converts each npy file into Pandas record
    '''

    mode = 'b'

    def load(self, f: IO) -> np.ndarray:
        ndarray = np.load(f)
        return {'ndarray': ndarray}

    def dump(self, obj: Dict[str, Any], f: IO) -> None:
        ndarray: np.ndarray = obj['ndarray']
        return np.save(f, ndarray)


class GetImageSizeFile(ItemStoreFileAdapter):
    mode = 'b'

    def load(self, f: IO) -> Dict[str, Tuple[int, int]]:
        image_size = get_image_size(f)
        return {'image_size': image_size}

    def dump(self, obj: Dict[str, Tuple[int, int]], f: IO) -> None:
        raise NotImplementedError


class COCOLabelsFile(ItemStoreFileAdapter):
    mode = 'b'

    def __init__(self, class_names: List[str], img_format: str):
        from cv_pipeliner.data_converters.coco import COCODataConverter
        self.coco_converter = COCODataConverter(class_names)
        self.img_format = img_format

    def load(self, f: fsspec.core.OpenFile) -> Dict[str, Tuple[int, int]]:
        filepath = pathlib(f.path)
        assert filepath.parent.name == 'labels'
        image_path = filepath.parent.parent / 'images' / f"{filepath.stem}.{self.img_format}"
        image_data = self.coco_converter.get_image_data_from_annot(
            image_path=f"{f.fs.protocol}://{image_path}", annot=f
        )
        return {'image_size': image_data}

    def dump(self, obj: Dict[str, Tuple[int, int]], f: fsspec.core.OpenFile) -> None:
        image_data: ImageData = obj['image_data']
        coco_data = self.coco_converter.get_annot_from_image_data(image_data)
        f.write('\n'.join(coco_data).encode())
