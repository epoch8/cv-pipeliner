import json

from typing import Any, Dict, IO
from datapipe.store.filedir import ItemStoreFileAdapter
from cv_pipeliner.core.data import ImageData


class ImageDataFile(ItemStoreFileAdapter):
    '''
    Converts each ImageData file into Pandas record
    '''

    mode = 't'

    def load(self, f: IO) -> ImageData:
        image_data = ImageData.from_json(json.load(f))
        return {'image_data': image_data}

    def dump(self, obj: Dict[str, Any], f: IO) -> None:
        image_data: ImageData = obj['image_data']
        return json.dump(image_data.json(), f, indent=4, ensure_ascii=False)
