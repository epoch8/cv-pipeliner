import io
import os
import re
import struct
import fsspec

from typing import Tuple
import numpy as np
from PIL import Image


def _exif_transpose_and_get_image_size(image: Image.Image) -> Tuple[int, int]:
    """
    If an image has an EXIF Orientation tag, return a size of image that is
    transposed accordingly. Otherwise, return the size of image itself.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112)
    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        old_width, old_height = image.size
        image_zeros = Image.fromarray(np.zeros((old_height, old_width))).transpose(method)
        return image_zeros.size

    return image.size


def get_image_size(filepath: 'bytes_or_path', exif_transpose: bool = True) -> Tuple[int, int]:
    if isinstance(filepath, io.BytesIO):  # file-like object
        fhandle = filepath
    elif isinstance(filepath, fsspec.core.OpenFile):  # file-like object
        fhandle = filepath.__enter__()
    else:
        fhandle = fsspec.open(filepath, 'rb').__enter__()

    image = Image.open(fhandle)  # lazy loading, so it's ok to open
    size = image.size
    if exif_transpose:
        size = _exif_transpose_and_get_image_size(image)
    return size
