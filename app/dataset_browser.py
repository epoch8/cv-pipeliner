import os
from typing import Callable
from collections import Counter

import numpy as np

from cv_pipeliner.visualizers.core.image_data import visualize_image_data
from cv_pipeliner.utils.images_datas import get_image_data_filtered_by_labels, get_n_bboxes_data_filtered_by_labels
from cv_pipeliner.utils.images import get_label_to_base_label_image
from src.data import get_images_data_from_dir, get_label_to_description
from src.config import get_cfg_defaults
from src.visualization import illustrate_bboxes_data, illustrate_n_bboxes_data

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)


if 'CV_PIPELINER_APP_CONFIG' in os.environ:
    config_file = os.environ['CV_PIPELINER_APP_CONFIG']
else:
    st.warning(
        "Environment variable 'CV_PIPELINER_APP_CONFIG' was not found. Loading default config instead."
    )
    config_file = 'config.yaml'

cfg = get_cfg_defaults()
cfg.merge_from_file(config_file)
cfg.freeze()


images_dirs = [list(d)[0] for d in cfg.data.images_dirs]
image_dir_to_annotation_filenames = {
    image_dir: d[image_dir] for d, image_dir in zip(cfg.data.images_dirs, images_dirs)
}
images_dirs = [image_dir for image_dir in images_dirs if len(image_dir_to_annotation_filenames[image_dir]) > 0]

images_from = st.sidebar.selectbox(
    'Image from',
    options=images_dirs
)
annotation_filename = st.sidebar.selectbox(
    'Annotation filename',
    options=image_dir_to_annotation_filenames[images_from]
)
images_data, annotation_success = get_images_data_from_dir(
    images_annotation_type=cfg.data.images_annotation_type,
    images_dir=images_from,
    annotation_filename=annotation_filename
)

if not annotation_success:
    st.warning("Annotations for given folder weren't found!")

view = st.sidebar.radio(
    label='View',
    options=["detection", "classification"]
)

mode = st.sidebar.radio(
    label='Output bboxes',
    options=["many", "one-by-one"],
    index=1
)

images_data_captions = [
    f"[{i}] {image_data.image_path.name} [{len(image_data.bboxes_data)} bboxes]"
    for i, image_data in enumerate(images_data)
]


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_get_label_to_base_label_image(**kwargs) -> Callable[[str], np.ndarray]:
    return get_label_to_base_label_image(**kwargs)


label_to_base_label_image = cached_get_label_to_base_label_image(base_labels_images_dir=cfg.data.base_labels_images_dir)
label_to_description = get_label_to_description(label_to_description_dict=cfg.data.labels_decriptions)

if view == 'detection':
    st.markdown("Choose an image:")
    images_data_selected_caption = st.selectbox(
        label='Image',
        options=[None] + images_data_captions
    )
    if images_data_selected_caption is not None:
        image_data_index = images_data_captions.index(images_data_selected_caption)
        image_data = images_data[image_data_index]
        st.text(images_data_selected_caption)

        labels = [bbox_data.label for bbox_data in image_data.bboxes_data]
    else:
        image_data = None
        labels = None

elif view == 'classification':
    bboxes_data = [bbox_data for image_data in images_data for bbox_data in image_data.bboxes_data]
    randomize = st.sidebar.checkbox('Shuffle bboxes')
    if randomize:
        np.random.shuffle(bboxes_data)
    labels = [bbox_data.label for bbox_data in bboxes_data]
else:
    image_data = None
    bboxes_data = None

if view == 'detection':
    use_labels = st.sidebar.checkbox(
        'Write labels',
        value=True
    )
    draw_label_images = st.sidebar.checkbox(
        'Draw base labels images',
        value=False
    )
    draw_base_labels_with_given_label_to_base_label_image = (
        label_to_base_label_image if draw_label_images else None
    )

if labels is not None:
    class_names_counter = Counter(labels)
    class_names = sorted(set(labels), key=class_names_counter.get, reverse=True)
    classes_to_find_captions = [
        f"{class_name} [{class_names_counter[class_name]} items]"
        for class_name in class_names
    ]
    filter_by_labels = st.multiselect(
        label="Classes to find",
        options=classes_to_find_captions,
        default=[]
    )
    filter_by_labels = [
        class_names[classes_to_find_captions.index(chosen_class_name)]
        for chosen_class_name in filter_by_labels
    ]


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_visualize_image_data(**kwargs) -> np.ndarray:
    return visualize_image_data(**kwargs)


if view == 'detection' and image_data is not None:
    image_data = get_image_data_filtered_by_labels(
        image_data=image_data,
        filter_by_labels=filter_by_labels
    )
    image = cached_visualize_image_data(
        image_data=image_data,
        use_labels=use_labels,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
    )
    st.image(image=image, use_column_width=True)
    illustrate_bboxes_data(
        true_image_data=image_data,
        label_to_base_label_image=label_to_base_label_image,
        label_to_description=label_to_description,
        mode=mode,
        background_color_a=[0, 0, 0, 255],
        true_background_color_b=[0, 255, 0, 255],
        bbox_offset=100,
        draw_rectangle_with_color=[0, 255, 0]
    )
elif view == 'classification' and bboxes_data is not None:
    n_bboxes_data = get_n_bboxes_data_filtered_by_labels(
        n_bboxes_data=[bboxes_data],
        filter_by_labels=filter_by_labels
    )
    illustrate_n_bboxes_data(
        n_bboxes_data=n_bboxes_data,
        label_to_base_label_image=label_to_base_label_image,
        label_to_description=label_to_description,
        mode=mode,
        background_color_a=[0, 0, 0, 255],
        true_background_color_b=[0, 255, 0, 255],
        bbox_offset=100,
        draw_rectangle_with_color=[0, 255, 0]
    )
