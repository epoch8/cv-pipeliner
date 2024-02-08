import io
import fsspec

from typing import Tuple
import numpy as np
from PIL import Image
from cv_pipeliner.utils.images import EXIF_ORIENTATION_TO_METHOD


def _exif_transpose_and_get_image_size(image: Image.Image) -> Tuple[int, int]:
    exif = image.getexif()
    orientation = exif.get(0x0112)
    method = EXIF_ORIENTATION_TO_METHOD.get(orientation)
    if method is not None:
        old_width, old_height = image.size
        image_zeros = Image.fromarray(np.zeros((old_height, old_width))).transpose(method)
        return image_zeros.size
    return image.size


def get_image_size(filepath: "bytes_or_path", exif_transpose: bool = True) -> Tuple[int, int]:
    if isinstance(filepath, bytes):  # file-like object
        fhandle = io.BytesIO(filepath)
    elif isinstance(filepath, io.BytesIO):  # file-like object
        fhandle = filepath
    elif isinstance(filepath, fsspec.core.OpenFile):  # file-like object
        fhandle = filepath.__enter__()
    else:
        fhandle = fsspec.open(filepath, "rb").__enter__()

    image = Image.open(fhandle)  # lazy loading, so it's ok to open
    size = image.size
    if exif_transpose:
        size = _exif_transpose_and_get_image_size(image)
    return size
