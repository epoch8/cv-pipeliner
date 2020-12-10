import os
import json
import fsspec
from typing import Dict, List, Literal, Tuple
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.data_converters.brickit import BrickitDataConverter
from cv_pipeliner.visualizers.core.image_data import visualize_image_data
from cv_pipeliner.utils.images_datas import get_image_data_filtered_by_labels, get_n_bboxes_data_filtered_by_labels
from cv_pipeliner.utils.images import get_label_to_base_label_image, open_image

from cv_pipeliner.utils.data import get_label_to_description
from cv_pipeliner.utils.streamlit.data import get_images_data_from_dir
from cv_pipeliner.utils.streamlit.visualization import illustrate_bboxes_data, illustrate_n_bboxes_data

from apps.config import get_cfg_defaults, merge_cfg_from_file_fsspec

st.set_option('deprecation.showfileUploaderEncoding', False)


config_file = os.environ['CV_PIPELINER_APP_CONFIG']
cfg = get_cfg_defaults()
merge_cfg_from_file_fsspec(cfg, config_file)
cfg.freeze()

ann_class_names = os.environ['CV_PIPELINER_ANN_CLASS_NAMES']
with fsspec.open(ann_class_names, 'r') as src:
    ann_class_names = json.load(src)

images_dirs = [list(d)[0] for d in cfg.data.images_dirs]
image_dir_to_annotation_filepaths = {
    image_dir: d[image_dir] for d, image_dir in zip(cfg.data.images_dirs, images_dirs)
}
images_dirs = [image_dir for image_dir in images_dirs if len(image_dir_to_annotation_filepaths[image_dir]) > 0]
images_dirname_to_image_dir_paths = {
    f"../{Path(image_dir).parent.name}/{Path(image_dir).name}": image_dir for image_dir in images_dirs
}

images_from = st.sidebar.selectbox(
    'Image from',
    options=list(images_dirname_to_image_dir_paths)
)
images_from = images_dirname_to_image_dir_paths[images_from]
annotation_filepath = st.sidebar.selectbox(
    'Annotation filepath',
    options=image_dir_to_annotation_filepaths[images_from]
)
annotation_openfile = fsspec.open(annotation_filepath, 'r')
images_data, annotation_success = get_images_data_from_dir(
    images_annotation_type=cfg.data.images_annotation_type,
    images_dir=images_from,
    annotation_filepath=annotation_filepath,
    annotation_filepath_st_mode=annotation_openfile.fs.checksum(annotation_openfile.path)  # for annotation changes
)
bbox_data_to_image_data_index_and_bboxes_data_subindex = {
    f'{bbox_data.image_path}_{(bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)}': (index, subindex)
    for index, image_data in enumerate(images_data)
    for subindex, bbox_data in enumerate(image_data.bboxes_data)
}

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
    f"[{i}] {image_data.image_name} [{len(image_data.bboxes_data)} bboxes]"
    for i, image_data in enumerate(images_data)
]


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_get_label_to_base_label_image(**kwargs) -> Dict[str, np.ndarray]:
    return get_label_to_base_label_image(**kwargs)


label_to_base_label_image = cached_get_label_to_base_label_image(base_labels_images=cfg.data.base_labels_images)
label_to_description = get_label_to_description(label_to_description_dict=cfg.data.labels_decriptions)
label_to_category = get_label_to_description(
    label_to_description_dict=os.environ['CV_PIPELINER_LABEL_TO_CATEGORY'],
    default_description='No category'
)

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
    labels = [bbox_data.label for bbox_data in bboxes_data]
    randomize = st.sidebar.checkbox('Shuffle bboxes')
    if randomize:
        np.random.shuffle(bboxes_data)
else:
    image_data = None
    bboxes_data = None

if view == 'detection':
    st.sidebar.title("Visualization")
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
    class_names = class_names + sorted(list(set(ann_class_names) - set(class_names)))
else:
    class_names_counter = {}
    class_names = sorted(ann_class_names)

classes_to_find_captions = [
    f"[{class_names_counter.get(class_name, 0)} items] {class_name} [{label_to_description[class_name]}]"
    for class_name in class_names
]
classes_to_find_captions_no_items = [
    f"{class_name} [{label_to_description[class_name]}]"
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
categories_by_class_names = [label_to_category[class_name] for class_name in class_names]
categories_counter = Counter(categories_by_class_names)
categories = sorted([
    category
    for category in set(label_to_category.values())
    if categories_counter[category] > 0
])

@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_visualize_image_data(**kwargs) -> np.ndarray:
    return visualize_image_data(**kwargs)


if '2020_12_08_validation_v3_mini' in images_from:
    for image_data in images_data:
        xmin, ymin, xmax, ymax = eval(str(image_data.image_path).split('crop_')[1].split('.jp')[0])
        image_data_with_crop = ImageData(
            image_path=image_data.image_path,
            bboxes_data=[BboxData(image_path=image_data.image_path, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)]
        )
        image_data.image = cached_visualize_image_data(image_data=image_data_with_crop)


def change_annotation(
    bbox_data: BboxData
):
    bbox_key = f'{bbox_data.image_path}_{(bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)}'
    change_annotation = st.checkbox(
        f'Change annotation',
        key=bbox_key
    ) if cfg.data.images_annotation_type == 'brickit' else False
    if change_annotation:
        chosen_category = st.selectbox(
            label="Categories",
            options=categories,
            index=categories.index(label_to_category[bbox_data.label])
        )
        class_names_by_category = [
            class_name
            for class_name in class_names
            if label_to_category[class_name] == chosen_category
        ]
        classes_to_find_captions_by_category = [
            f"{class_name} [{label_to_description[class_name]}]"
            for class_name in class_names_by_category
        ]
        new_label_caption = st.selectbox(
            label="Classes",
            options=classes_to_find_captions_by_category,
            index=class_names_by_category.index(bbox_data.label) if (
                bbox_data.label in class_names_by_category
            ) else 0
        )
        new_label = class_names_by_category[classes_to_find_captions_by_category.index(new_label_caption)]
        st.image(image=label_to_base_label_image[new_label])
        save_button = st.button('Save annotation', key=bbox_key)
        if save_button:
            index, subindex = bbox_data_to_image_data_index_and_bboxes_data_subindex[bbox_key]
            images_data[index].bboxes_data[subindex].label = new_label
            new_annotation = BrickitDataConverter().get_annot_from_images_data(images_data)
            with fsspec.open(annotation_filepath, 'w') as out:
                json.dump(new_annotation, out, indent=4)
            st.text('Success! Annotation is updated. Rerun to see changes.')


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
        draw_rectangle_with_color=[0, 255, 0],
        change_annotation=change_annotation
    )
