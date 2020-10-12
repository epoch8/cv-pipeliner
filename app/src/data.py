from pathlib import Path
from typing import Union, Literal, List

from cv_pipeliner.core.data import ImageData

from cv_pipeliner.data_converters.supervisely import SuperviselyDataConverter
from cv_pipeliner.data_converters.brickit import BrickitDataConverter

import streamlit as st


@st.cache(show_spinner=False)
def get_images_data_from_dir(
    images_annotation_type: Literal['brickit', 'supervisely'],
    images_dir: Union[str, Path],
    annotation_filename: str = None
) -> List[ImageData]:
    images_dir = Path(images_dir)
    image_paths = sorted(
        list(images_dir.glob('*.png')) + list(images_dir.glob('*.jp*g'))
    )
    annotation_success = False
    if images_annotation_type == 'brickit':
        if annotation_filename is None:
            annotation_filename = 'annotations.json'
        else:
            annots = images_dir / annotation_filename
        if annots.exists():
            images_data = BrickitDataConverter().get_images_data_from_annots(
                image_paths=image_paths, annots=annots
            )
            annotation_success = True

    elif images_annotation_type == 'supervisely':
        annots_paths = sorted(
            annot_path for annot_path in images_dir.glob('*.json')
            if annot_path != 'meta.json'
        )
        if len(image_paths) == len(annots_paths):
            images_data = SuperviselyDataConverter().get_images_data_from_annots(
                image_paths=image_paths, annots=annots_paths
            )
        else:
            images_data = [ImageData(image_path=image_path) for image_path in image_paths]

    if not annotation_success:
        images_data = [ImageData(image_path=image_path) for image_path in image_paths]

    images_data = sorted(images_data,
                         key=lambda image_data: len(image_data.bboxes_data),
                         reverse=True)

    return images_data, annotation_success


@st.cache(show_spinner=False)
def get_videos_data_from_dir(
    videos_dir: Union[str, Path]
) -> List[Path]:
    videos_dir = Path(videos_dir)
    videos_paths = sorted(
        list(videos_dir.glob('*.mp4')) + list(videos_dir.glob('*.m4v')) + list(videos_dir.glob('*.mov'))
    )
    return videos_paths
