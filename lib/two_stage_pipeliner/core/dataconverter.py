
import abc
from typing import Union, List
from pathlib import Path
from two_stage_pipeliner.core.data import ImageData, BboxData


class DataConverter(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_image_data(self,
                       image_path: Union[Path, str],
                       annot_path: Union[Path, str]) -> List[ImageData]:
        pass

    @abc.abstractmethod
    def get_bbox_data(self,
                      image_path: Union[Path, str],
                      annot_path: Union[Path, str]) -> List[BboxData]:
        pass
