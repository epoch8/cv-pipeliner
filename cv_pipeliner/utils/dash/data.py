from typing import Union, Literal, List
from pathlib import Path
from pathy import Pathy
from cv_pipeliner.core.data import ImageData

from cv_pipeliner.data_converters.supervisely import SuperviselyDataConverter
from cv_pipeliner.data_converters.brickit import BrickitDataConverter

import fsspec


def get_images_data_from_dir(
    images_annotation_type: Literal['brickit', 'supervisely'],
    images_dir: Union[str, Path],
    annotation_filepath: Union[str, Path, None]
) -> List[ImageData]:
    images_dir = Pathy(images_dir)
    image_paths = sorted(
        fsspec.open_files(str(images_dir / '*.png')) +
        fsspec.open_files(str(images_dir / '*.jp*g')),
        key=lambda f: f.path
    )
    annotation_success = False

    if annotation_filepath is not None:
        annotation_filepath = Pathy(annotation_filepath)
        if images_annotation_type == 'brickit':
            images_data = BrickitDataConverter().get_images_data_from_annots(
                image_paths=image_paths,
                annots=annotation_filepath
            )
            annotation_success = True

        elif images_annotation_type == 'supervisely':
            annots_paths = sorted(
                fsspec.open_files(str(annotation_filepath / '*.json')),
                key=lambda f: f.path
            )
            if len(image_paths) == len(annots_paths):
                images_data = SuperviselyDataConverter().get_images_data_from_annots(
                    image_paths=image_paths,
                    annots=annots_paths
                )
                annotation_success = True
            else:
                raise ValueError('Supervisely annotation: len(image_paths) != len(annots_paths).')

    if not annotation_success:
        images_data = [ImageData(image_path=image_path) for image_path in image_paths]

    return images_data, annotation_success
