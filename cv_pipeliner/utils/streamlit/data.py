import json

from typing import Union, Literal, List, Callable, Dict
from pathlib import Path
from pathy import Pathy
from cv_pipeliner.core.data import ImageData

from cv_pipeliner.data_converters.supervisely import SuperviselyDataConverter
from cv_pipeliner.data_converters.brickit import BrickitDataConverter

import streamlit as st
import fsspec

from cv_pipeliner.utils.files import fixed_fsspec_glob


@st.cache(show_spinner=False)
def get_images_data_from_dir(
    images_annotation_type: Literal['brickit', 'supervisely'],
    images_dir: Union[str, Path],
    annotation_filepath: Union[str, Path] = None,
    fs: fsspec.filesystem = fsspec.filesystem('file')
) -> List[ImageData]:
    images_dir = Pathy(images_dir)
    image_paths = sorted(
        list(fixed_fsspec_glob(fs, str(images_dir / '*.png'))) + list(fixed_fsspec_glob(fs, str(images_dir / '*.jp*g')))
    )
    annotation_filepath = Pathy(annotation_filepath)
    annotation_success = False
    if images_annotation_type == 'brickit':
        if annotation_filepath is None:
            annotation_filepath = images_dir / 'annotations.json'
        if fs.exists(annotation_filepath):
            images_data = BrickitDataConverter().get_images_data_from_annots(
                image_paths=image_paths,
                annots=annotation_filepath,
                fs=fs
            )
            annotation_success = True

    elif images_annotation_type == 'supervisely':
        annots_paths = sorted(
            annot_path for annot_path in fixed_fsspec_glob(fs, str(annotation_filepath / '*.json'))
            if annot_path != 'meta.json'
        )
        if len(image_paths) == len(annots_paths):
            images_data = SuperviselyDataConverter().get_images_data_from_annots(
                image_paths=image_paths,
                annots=annots_paths,
                fs=fs
            )
            annotation_success = True
        else:
            images_data = [ImageData(image_path=image_path) for image_path in image_paths]

    if not annotation_success:
        images_data = [ImageData(image_path=image_path) for image_path in image_paths]

    images_data = sorted(images_data,
                         key=lambda image_data: len(image_data.bboxes_data),
                         reverse=True)

    return images_data, annotation_success


@st.cache(show_spinner=False)
def get_label_to_description(
    label_to_description_dict: Union[str, Path, Dict],
    fs: fsspec.filesystem = fsspec.filesystem('file')
) -> Callable[[str], str]:
    if isinstance(label_to_description_dict, str) or isinstance(label_to_description_dict, Path):
        with fs.open(label_to_description_dict, 'r') as src:
            label_to_description_dict = json.load(src)

    label_to_description_dict['unknown'] = 'No description.'

    def label_to_description(label: str) -> str:
        if label in label_to_description_dict:
            return label_to_description_dict[label]
        else:
            return label_to_description_dict['unknown']

    return label_to_description
