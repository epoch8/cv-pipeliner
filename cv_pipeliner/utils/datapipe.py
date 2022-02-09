import json
import numpy as np
from typing import Any, Dict, IO

from datapipe.store.filedir import ItemStoreFileAdapter
from cv_pipeliner.core.data import ImageData


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
