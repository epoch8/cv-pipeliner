from typing import Callable, Iterable, List, Optional, Type, TypeVar, Union

from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.core.data import BboxData, ImageData

T = TypeVar("T")


def split_chunks(items: List[T], shapes: Iterable[int]) -> List[List[T]]:
    chunks = []
    count = 0
    for shape in shapes:
        chunks.append(items[count : count + shape])
        count += shape
    return chunks


def call_progress_callback(progress_callback: Optional[Callable[[int], None]], progress: int) -> None:
    if progress_callback is not None:
        progress_callback(progress)


def ensure_image_data_generator(
    data: Union[List[ImageData], BatchGeneratorImageData],
    batch_size_default: int,
) -> BatchGeneratorImageData:
    if isinstance(data, list):
        return BatchGeneratorImageData(data, batch_size=batch_size_default)
    if isinstance(data, BatchGeneratorImageData):
        return data
    raise TypeError(f"Unknown type of images data generator: {type(data)}")


def ensure_bbox_data_generator(
    data: Union[List[BboxData], List[List[BboxData]], BatchGeneratorBboxData],
    batch_size_default: int,
) -> BatchGeneratorBboxData:
    if isinstance(data, BatchGeneratorBboxData):
        return data
    if isinstance(data, list):
        if all(isinstance(item, BboxData) for item in data):
            data = [data]
        return BatchGeneratorBboxData(data, batch_size=batch_size_default)
    raise TypeError(f"Unknown type of bboxes data generator: {type(data)}")


def ensure_image_or_bbox_data_generator(
    data: Union[List[ImageData], List[BboxData], List[List[BboxData]], BatchGeneratorImageData, BatchGeneratorBboxData],
    batch_size_default: int,
) -> Union[BatchGeneratorImageData, BatchGeneratorBboxData]:
    if isinstance(data, (BatchGeneratorImageData, BatchGeneratorBboxData)):
        return data
    if isinstance(data, list):
        if all(isinstance(item, ImageData) for item in data):
            return BatchGeneratorImageData(data, batch_size=batch_size_default)
        if all(isinstance(item, BboxData) for item in data) or all(
            isinstance(item, (list, tuple)) for item in data
        ):
            return ensure_bbox_data_generator(data, batch_size_default=batch_size_default)
    raise TypeError(f"Unknown type of data generator: {type(data)}")


def require_instance(value, cls: Type[T], name: str) -> T:
    assert isinstance(value, cls), f"{name} should be {cls.__name__}, got {type(value).__name__}"
    return value
