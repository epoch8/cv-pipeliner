import os
import datetime
import json
import fsspec
import copy
import time
import re
from typing import Dict, List, Literal, Tuple
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from pathy import Pathy

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.data_converters.brickit import BrickitDataConverter
from cv_pipeliner.visualizers.core.image_data import visualize_image_data
from cv_pipeliner.utils.images_datas import get_image_data_filtered_by_labels, get_n_bboxes_data_filtered_by_labels
from cv_pipeliner.utils.images import (
    get_label_to_base_label_image, open_image,
    draw_n_base_labels_images
)
from cv_pipeliner.utils.data import get_label_to_description
from cv_pipeliner.utils.streamlit.data import get_images_data_from_dir
from cv_pipeliner.utils.streamlit.visualization import (
    get_illustrated_bboxes_data,
    illustrate_bboxes_data, illustrate_n_bboxes_data, fetch_page_session
)
from apps.config import get_cfg_defaults, merge_cfg_from_file_fsspec

st.set_option('deprecation.showfileUploaderEncoding', False)


config_file = os.environ['CV_PIPELINER_APP_CONFIG']
cfg = get_cfg_defaults()
merge_cfg_from_file_fsspec(cfg, config_file)
cfg.freeze()

with fsspec.open(cfg.data.ann_class_names, 'r') as src:
    ann_class_names = json.load(src)

images_dirs = [list(d)[0] for d in cfg.data.images_dirs]
image_dir_to_annotation_filepaths = {
    image_dir: d[image_dir] for d, image_dir in zip(cfg.data.images_dirs, images_dirs)
}
images_dirs = [image_dir for image_dir in images_dirs if len(image_dir_to_annotation_filepaths[image_dir]) > 0]

images_from = st.sidebar.selectbox(
    'Image from',
    options=list(images_dirs),
    format_func=lambda image_dir: f"../{Pathy(image_dir).name}"
)
st.sidebar.markdown(f'Images from: {images_from}')
annotation_filepath = st.sidebar.selectbox(
    'Annotation filepath',
    options=image_dir_to_annotation_filepaths[images_from],
    format_func=lambda filepath: f"../{Pathy(filepath).name}"
)
st.sidebar.markdown(f'Annotation: {annotation_filepath}')
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

annotation_mode = st.sidebar.checkbox(
    label='Annotation mode',
    value=False
) if cfg.data.images_annotation_type == 'brickit' else None

images_data_captions = [
    f"[{i}] {image_data.image_name} [{len(image_data.bboxes_data)} bboxes]"
    for i, image_data in enumerate(images_data)
]


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_get_label_to_base_label_image(**kwargs) -> Dict[str, np.ndarray]:
    return get_label_to_base_label_image(**kwargs)

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

label_to_base_label_image = cached_get_label_to_base_label_image(base_labels_images=cfg.data.base_labels_images)
label_to_description = get_label_to_description(label_to_description_dict=cfg.data.labels_decriptions)
label_to_category = get_label_to_description(
    label_to_description_dict=cfg.data.label_to_category,
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
else:
    image_data = None
    bboxes_data = None

st.sidebar.markdown('---')
if not annotation_mode:
    average_maximum_images_per_page = st.sidebar.slider(
        label='Maximum images per page',
        min_value=1,
        max_value=100,
        value=50
    )
else:
    average_maximum_images_per_page = 1

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

class_names_counter = Counter(labels) if labels is not None else {}
class_names = sorted(
    ann_class_names,
    key=lambda x: int(re.sub('\D', '', x)) if re.sub('\D', '', x).isdigit() else 0
)

classes_col1, classes_col2 = st.beta_columns(2)
with classes_col1:
    label_to_description['non_default_class'] = "Not from ann_class_names.json"
    format_func_filter = lambda class_name: (
        f"{class_name} [{label_to_description[class_name]}]"
    )
    filter_by_labels = st.multiselect(
        label="Classes to find",
        options=['non_default_class'] + class_names,
        default=[],
        format_func=format_func_filter
    )
    st.markdown(f'Classes chosen: {", ".join([format_func_filter(class_name) for class_name in filter_by_labels])}')
with classes_col2:
    sorted_class_names = sorted(
        class_names, key=lambda class_name: class_names_counter.get(class_name, 0), reverse=True
    )
    show_df = st.checkbox(
        label='Show count df'
    )
    if show_df:
        df = pd.DataFrame({
            'class_name': sorted_class_names,
            'count': list(map(lambda class_name: class_names_counter.get(class_name, 0), sorted_class_names))
        })
        st.dataframe(data=df, width=1000)
filter_by_labels = [
    chosen_class_name
    for chosen_class_name in filter_by_labels
]
if 'non_default_class' in filter_by_labels and labels is not None:
    filter_by_labels = sorted(set(labels) - set(ann_class_names))
    if len(filter_by_labels) == 0:
        filter_by_labels = ['non_default_class']
categories_by_class_names = [label_to_category[class_name] for class_name in class_names]
categories_counter = Counter(categories_by_class_names)
categories = sorted([
    category
    for category in set(label_to_category.values())
    if categories_counter[category] > 0
])

def change_annotation(
    bbox_data: BboxData,
    max_page: int
):
    global bbox_data_to_image_data_index_and_bboxes_data_subindex, images_data, annotation_success

    bbox_key = f'{bbox_data.image_path}_{(bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax)}'
    _, del_col, _, _ = st.beta_columns(4)
    if annotation_mode:
        change_annotation = True
        col1, col2 = st.beta_columns(2)
        with col1:
            prev_button = st.button('Previous')
        with col2:
            next_button = st.button('Next')
        if prev_button:
            page_session = fetch_page_session()
            page_session.counter -= 1
            page_session.counter = max_page if page_session.counter == 0 else page_session.counter
            st.experimental_rerun()
        if next_button:
            page_session = fetch_page_session()
            page_session.counter += 1
            page_session.counter = 1 if page_session.counter > max_page else page_session.counter
            st.experimental_rerun()
        st.markdown('---')
    else:
        change_annotation = st.checkbox(
            f'Change annotation',
            value=annotation_mode,
            key=bbox_key
        ) if cfg.data.images_annotation_type == 'brickit' else False
    if change_annotation:
        with del_col:
            delete_checkbox = st.checkbox('Delete this bbox', key=bbox_key)
            delete_button = st.button('Delete this bbox (confirm)', key=bbox_key) if delete_checkbox else False
        bbox_input = st.text_input(
            label='Bbox',
            value=f"{[bbox_data.xmin, bbox_data.ymin, bbox_data.xmax, bbox_data.ymax]}"
        )
        col1, col2 = st.beta_columns(2)
        _, col3, _ = st.beta_columns(3)
        xmin, ymin, xmax, ymax = eval(bbox_input)
        if bbox_data.top_n is not None and bbox_data.top_n > 1:
            for i in range(14):
                st.text('\n')
            top_n_hint_image = draw_n_base_labels_images(
                labels=bbox_data.labels_top_n,
                label_to_base_label_image=label_to_base_label_image,
                label_to_description=label_to_description
            )
            st.image(top_n_hint_image, use_column_width=True)
        with col1:
            chosen_category = st.selectbox(
                label="Categories",
                options=['All', 'Custom'] + categories,
                index=2+categories.index(label_to_category[bbox_data.label]) if not annotation_mode else 0,
                key=f'category_{bbox_key}'
            )
            if chosen_category != 'Custom':
                class_names_by_category = [
                    class_name
                    for class_name in class_names
                    if label_to_category[class_name] == chosen_category or chosen_category == 'All'
                ]
                if bbox_data.top_n is not None and bbox_data.top_n > 1:
                    class_names_by_category = bbox_data.labels_top_n + class_names_by_category
                def format_func(class_name):
                    if bbox_data.top_n is not None and bbox_data.top_n > 1 and class_name in bbox_data.labels_top_n:
                        index = bbox_data.labels_top_n.index(class_name)
                        return f"[{index+1}] {class_name} [{label_to_description[class_name]}]"
                    else:
                        return f"{class_name} [{label_to_description[class_name]}]"
                new_label = st.selectbox(
                    label="Classes",
                    options=class_names_by_category,
                    index=class_names_by_category.index(bbox_data.label) if (
                        bbox_data.label in class_names_by_category
                    ) else 0,
                    key=f'classes_{bbox_key}',
                    format_func=format_func
                )
            else:
                new_label = st.text_input(
                    label='Write your own class name'
                )
        with col2:
            new_bbox_data = copy.deepcopy(bbox_data)
            new_bbox_data.xmin = xmin
            new_bbox_data.ymin = ymin
            new_bbox_data.xmax = xmax
            new_bbox_data.ymax = ymax
            new_bbox_data.label = new_label
            index, _ = bbox_data_to_image_data_index_and_bboxes_data_subindex[bbox_key]
            cropped_images_and_renders, _ = get_illustrated_bboxes_data(
                source_image=images_data[index].open_image(),
                bboxes_data=[new_bbox_data],
                label_to_base_label_image=label_to_base_label_image,
                background_color_a=[0, 0, 0, 255],
                true_background_color_b=[0, 255, 0, 255],
                bbox_offset=50,
                draw_rectangle_with_color=[0, 255, 0]
            )
            st.image(image=cropped_images_and_renders[0], use_column_width=True)
        with col3:
            update_button = st.button('Update', key=bbox_key)
        if update_button or delete_button:

            # Scenary when 2 people are in: create the lock file of annotation
            locker_filepath = f"{annotation_filepath}.lock"
            temp_file_lock_r = fsspec.open(locker_filepath, 'r')
            with st.spinner(
                'Locker is on (someone is changing annotation too). Please wait...\n'
                f'If this is deadlock, delete this file: {locker_filepath}'
            ):
                while temp_file_lock_r.fs.exists(locker_filepath):
                    time.sleep(0.1)
            temp_file_lock = fsspec.open(locker_filepath, 'w')
            with temp_file_lock as out:
                out.write('Lock')

            # We need reread annotation (scenary when 2 people are in):
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
            index, subindex = bbox_data_to_image_data_index_and_bboxes_data_subindex[bbox_key]
            if update_button:
                images_data[index].bboxes_data[subindex].xmin = xmin
                images_data[index].bboxes_data[subindex].ymin = ymin
                images_data[index].bboxes_data[subindex].xmax = xmax
                images_data[index].bboxes_data[subindex].ymax = ymax
                images_data[index].bboxes_data[subindex].label = new_label
            elif delete_button:
                del images_data[index].bboxes_data[subindex]
            new_annotation = BrickitDataConverter().get_annot_from_images_data(images_data)

            # create backup
            annotation_fileopen = fsspec.open(annotation_filepath, 'r')
            now = datetime.datetime.now().strftime('%Y_%m_%d_%Hh')
            backup_filepath = Pathy(annotation_filepath).parent / f'{Pathy(annotation_filepath).name}.{now}_backup'
            if not annotation_fileopen.fs.exists(str(backup_filepath)):
                with fsspec.open(annotation_filepath, 'r') as src:
                    with fsspec.open(str(backup_filepath), 'w') as out:
                        out.write(src.read())

            # update annotation file
            with fsspec.open(annotation_filepath, 'w') as out:
                json.dump(new_annotation, out)

            # delete lock file
            temp_file_lock.fs.rm(locker_filepath)

            # rerun
            st.experimental_rerun()


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
        mode='one-by-one',
        background_color_a=[0, 0, 0, 255],
        true_background_color_b=[0, 255, 0, 255],
        bbox_offset=100,
        draw_rectangle_with_color=[0, 255, 0],
        change_annotation=change_annotation,
        average_maximum_images_per_page=average_maximum_images_per_page
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
        mode='one-by-one',
        background_color_a=[0, 0, 0, 255],
        true_background_color_b=[0, 255, 0, 255],
        bbox_offset=100,
        draw_rectangle_with_color=[0, 255, 0],
        change_annotation=change_annotation,
        average_maximum_images_per_page=average_maximum_images_per_page
    )
elif view == 'annotation':
    n_bboxes_data = get_n_bboxes_data_filtered_by_labels(
        n_bboxes_data=[bboxes_data],
        filter_by_labels=filter_by_labels
    )
    illustrate_n_bboxes_data(
        n_bboxes_data=n_bboxes_data,
        label_to_base_label_image=label_to_base_label_image,
        label_to_description=label_to_description,
        mode='one-by-one',
        background_color_a=[0, 0, 0, 255],
        true_background_color_b=[0, 255, 0, 255],
        bbox_offset=100,
        draw_rectangle_with_color=[0, 255, 0],
        change_annotation=change_annotation,
        average_maximum_images_per_page=average_maximum_images_per_page
    )
