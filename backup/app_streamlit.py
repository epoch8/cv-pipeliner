import os
import json
from collections import Counter
from pathlib import Path
from typing import Dict
from io import BytesIO
from urllib.parse import urljoin

import requests
import numpy as np
import pandas as pd
import fsspec

from pathy import Pathy
from dacite import from_dict
from PIL import Image

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.visualizers.core.image_data import visualize_image_data
from cv_pipeliner.utils.images_datas import get_image_data_filtered_by_labels
from cv_pipeliner.utils.images import get_label_to_base_label_image

import streamlit as st
from cv_pipeliner.utils.data import get_label_to_description
from cv_pipeliner.utils.streamlit.data import get_images_data_from_dir
from cv_pipeliner.utils.streamlit.visualization import illustrate_bboxes_data, fetch_image_data
from cv_pipeliner.utils.models_definitions import DetectionModelDefinition, ClassificationModelDefinition

from apps.config import get_cfg_defaults, merge_cfg_from_file_fsspec


st.set_option('deprecation.showfileUploaderEncoding', False)

config_file = os.environ['CV_PIPELINER_APP_CONFIG']
frontend_url = os.environ['CV_PIPELINER_FRONTEND_URL']

cfg = get_cfg_defaults()
merge_cfg_from_file_fsspec(cfg, config_file)
cfg.freeze()


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_get_label_to_base_label_image(**kwargs) -> Dict[str, str]:
    return get_label_to_base_label_image(**kwargs)


label_to_base_label_image = cached_get_label_to_base_label_image(base_labels_images=cfg.data.base_labels_images)
label_to_description = get_label_to_description(label_to_description_dict=cfg.data.labels_decriptions)
models_definitions_response = requests.get(urljoin(backend_url, 'get_available_models/'))
if models_definitions_response.ok:
    models_definitions = json.loads(models_definitions_response.text)
    detection_models_definitions = [
        from_dict(data_class=DetectionModelDefinition, data=detection_model_definition)
        for detection_model_definition in models_definitions['detection_models_definitions']
    ]
    classification_models_definitions = [
        from_dict(data_class=ClassificationModelDefinition, data=classification_model_definition)
        for classification_model_definition in models_definitions['classification_models_definitions']
    ]
else:
    raise ValueError(
        f'Something wrong with backend. Response: {models_definitions_response.text}'
    )
description_to_detection_model_definition = {
    detection_model_definition.description: detection_model_definition
    for detection_model_definition in detection_models_definitions
}
description_to_classiticaion_model_definition = {
    classification_model_definition.description: classification_model_definition
    for classification_model_definition in classification_models_definitions
}
current_model_definition_response = requests.get(urljoin(backend_url, 'get_current_models/'))
if current_model_definition_response.ok:
    current_models_definitions = json.loads(current_model_definition_response.text)
    current_detection_model_definition = from_dict(
        data_class=DetectionModelDefinition,
        data=current_models_definitions['detection_model_definition']
    )
    current_classification_model_definition = from_dict(
        data_class=ClassificationModelDefinition,
        data=current_models_definitions['classification_model_definition']
    )
else:
    raise ValueError(
        f'Something wrong with backend. Response: {models_definitions_response.text}'
    )

st.sidebar.title("Pipeline")
st.sidebar.header('Detection')
detection_descriptions = [description for description in description_to_detection_model_definition]
detection_model_description = st.sidebar.selectbox(
    label='Model:',
    options=detection_descriptions,
    index=detection_descriptions.index(current_detection_model_definition.description)
)
st.sidebar.markdown(f'Chosen detection model: {detection_model_description}')
st.sidebar.header('Classification')
classification_descriptions = [description for description in description_to_classiticaion_model_definition]
classification_model_description = st.sidebar.selectbox(
    label='Model:',
    options=classification_descriptions,
    index=classification_descriptions.index(current_classification_model_definition.description)
)
st.sidebar.markdown(f'Chosen classification model: {classification_model_description}')
detection_model_definition = description_to_detection_model_definition[detection_model_description]
classification_model_definition = description_to_classiticaion_model_definition[classification_model_description]

if detection_model_definition.model_index != current_detection_model_definition.model_index:
    requests.post(urljoin(backend_url, f'set_detection_model/{detection_model_definition.model_index}'))
if classification_model_definition.model_index != current_classification_model_definition.model_index:
    requests.post(urljoin(backend_url, f'set_classification_model/{classification_model_definition.model_index}'))

st.sidebar.subheader('Detection score threshold')
detection_score_threshold = st.sidebar.slider(
    label='Threshold',
    min_value=0.,
    max_value=1.,
    value=detection_model_definition.score_threshold,
    step=0.05
)
if (
    isinstance(classification_model_definition.model_spec.class_names, str)
    or
    isinstance(classification_model_definition.model_spec.class_names, Path)
):
    with fsspec.open(classification_model_definition.model_spec.class_names, 'r') as src:
        class_names = json.load(src)
else:
    class_names = classification_model_definition.model_spec.class_names
class_names = sorted(
    set(class_names),
    key=lambda x: int(x) if x.isdigit() else 0
)
add_description_to_class_names = st.sidebar.checkbox('Add description in classes list')
if add_description_to_class_names:
    classes_to_find_captions = [
        f"{class_name} [{label_to_description[class_name]}]"
        for class_name in class_names
    ]
else:
    classes_to_find_captions = class_names
filter_by_labels = st.sidebar.multiselect(
    label="Classes to find",
    options=classes_to_find_captions,
    default=[]
)
hide_labels = st.sidebar.multiselect(
    label="Classes to hide",
    options=classes_to_find_captions,
    default=[]
)
if len(hide_labels) != 0:
    filter_by_labels = [
        class_names[classes_to_find_captions.index(chosen_class_name)]
        for chosen_class_name in filter_by_labels
    ] if len(filter_by_labels) > 0 else class_names
    hide_labels = [
        class_names[classes_to_find_captions.index(chosen_class_name)]
        for chosen_class_name in hide_labels
    ]
filter_by_labels = list(set(filter_by_labels) - set(hide_labels))

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

input_type = st.radio(
    label='Input',
    options=["Image", "Camera"]
)

if input_type == 'Image':
    images_dirs = [list(d)[0] for d in cfg.data.images_dirs]
    image_dir_to_annotation_filepaths = {
        image_dir: d[image_dir] for d, image_dir in zip(cfg.data.images_dirs, images_dirs)
    }

    images_from = st.selectbox(
        'Image from',
        options=['Upload'] + list(images_dirs),
        format_func=lambda image_dir: f"../{Pathy(image_dir).name}"
    )

    if images_from == 'Upload':
        st.header('Image')
        image_bytes = st.file_uploader("Upload image", type=["png", "jpeg", "jpg"])
        if image_bytes is not None:
            image_data = ImageData(image_path=image_bytes.getvalue())
        else:
            image_data = None
        show_annotation = False
    else:
        annotation_filepath = st.selectbox(
            'Annotation filepath',
            options=image_dir_to_annotation_filepaths[images_from],
            format_func=lambda filepath: f"../{Pathy(filepath).name}"
        )
        images_data, annotation_success = get_images_data_from_dir(
            images_annotation_type=cfg.data.images_annotation_type,
            images_dir=images_from,
            annotation_filepath=annotation_filepath
        )
        if annotation_success:
            images_data_captions = [
                f"[{i}] {image_data.image_name} [{len(image_data.bboxes_data)} bboxes]"
                for i, image_data in enumerate(images_data)
            ]
        else:
            images_data_captions = [
                f"[{i}] {image_data.image_name}"
                for i, image_data in enumerate(images_data)
            ]
        images_data_selected_caption = st.selectbox(
            label='Image',
            options=[None] + images_data_captions
        )
        if images_data_selected_caption is not None:
            image_data_index = images_data_captions.index(images_data_selected_caption)
            image_data = images_data[image_data_index]
            st.text(images_data_selected_caption)
        else:
            image_data = None
        if annotation_success:
            show_annotation = st.checkbox('Show annotation', value=False)
        else:
            show_annotation = False

    mode = st.sidebar.radio(
        label='Output bboxes',
        options=["many", "one-by-one"],
        index=1
    )
    show_top_n = st.sidebar.checkbox(
        label="Show classification's top-n labels",
        value=False
    )
    average_maximum_images_per_page = st.sidebar.slider(
        label='Maximum images per page',
        min_value=1,
        max_value=100,
        value=50
    )
    if show_top_n:
        classification_top_n = st.sidebar.slider(
            label='Top-n',
            min_value=1,
            max_value=20,
            value=5
        )
    else:
        classification_top_n = 1
elif input_type == "Camera":
    st.markdown(
        f"# Check your model settings and go to the next url: {frontend_url}"
    )


@st.cache(show_spinner=False, allow_output_mutation=True)
def inference_one_image(
    backend_url: str,
    detection_model_index: str,
    classification_model_index: str,
    image_data: ImageData,
    detection_score_threshold: float,
    classification_top_n: int
) -> ImageData:
    url_post = urljoin(
        backend_url,
        (
            'predict'
            f'?detection_model_index={detection_model_index}&'
            f'classification_model_index={classification_model_index}&'
            f'detection_score_threshold={detection_score_threshold}&'
            f'classification_top_n={classification_top_n}'
        )
    )
    image = Image.fromarray(image_data.open_image())
    image_bytes = BytesIO()
    image.save(image_bytes, format='jpeg', quality=100)
    image_bytes = image_bytes.getvalue()
    response = requests.post(url_post, files={'image': image_bytes})
    if response.ok:
        pred_image_data = ImageData.from_dict(json.loads(response.text))
        pred_image_data.image_path = image_data.image_path
    else:
        raise ValueError(
            f'Something wrong with backend. Response: {response.text}'
        )

    return pred_image_data


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_visualize_image_data(**kwargs) -> np.ndarray:
    return visualize_image_data(**kwargs)


if input_type == 'Image':
    run = st.checkbox('RUN PIPELINE')
    if run and image_data is not None:
        with st.spinner("Working on your image..."):
            pred_image_data = inference_one_image(
                backend_url=backend_url,
                detection_model_index=detection_model_definition.model_index,
                classification_model_index=classification_model_definition.model_index,
                image_data=image_data,
                detection_score_threshold=detection_score_threshold,
                classification_top_n=classification_top_n
            )
        image_data = get_image_data_filtered_by_labels(
            image_data=image_data,
            filter_by_labels=filter_by_labels
        )
        pred_image_data = get_image_data_filtered_by_labels(
            image_data=pred_image_data,
            filter_by_labels=filter_by_labels
        )
        pred_image = cached_visualize_image_data(
            image_data=pred_image_data,
            use_labels=use_labels,
            draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
        )
        st.image(image=pred_image, use_column_width=True)
        if show_annotation:
            illustrate_bboxes_data(
                true_image_data=image_data,
                label_to_base_label_image=label_to_base_label_image,
                label_to_description=label_to_description,
                mode=mode,
                pred_image_data=pred_image_data,
                minimum_iou=cfg.data.minimum_iou,
                background_color_a=[0, 0, 0, 255],
                true_background_color_b=[0, 255, 0, 255],
                pred_background_color_b=[255, 255, 0, 255],
                bbox_offset=100,
                draw_rectangle_with_color=[0, 255, 0],
                show_top_n=show_top_n,
                average_maximum_images_per_page=average_maximum_images_per_page
            )
        else:
            fast_annotation_mode = st.checkbox(
                label="Annotate Correct/Incorrect bboxes",
                value=False
            ) if run and mode == "one-by-one" and not show_annotation else False
            if fast_annotation_mode:
                col1, col2 = st.beta_columns((3, 2))
                with col1:
                    df_errors_placeholder = st.empty()
                with col2:
                    df_accuracy_placeholder = st.empty()
            illustrate_bboxes_data(
                true_image_data=pred_image_data,
                label_to_base_label_image=label_to_base_label_image,
                label_to_description=label_to_description,
                mode=mode,
                background_color_a=[0, 0, 0, 255],
                true_background_color_b=[255, 255, 0, 255],
                bbox_offset=100,
                draw_rectangle_with_color=[0, 255, 0],
                show_top_n=show_top_n,
                fast_annotation_mode=fast_annotation_mode,
                average_maximum_images_per_page=average_maximum_images_per_page
            )
            if fast_annotation_mode:
                page_session = fetch_image_data(image_data=pred_image_data)
                errors_counter = Counter(
                    [value for value in page_session.bboxes_data_fast_annotations.values() if value is not None]
                )
                total = sum(errors_counter.values())
                df_errors = pd.DataFrame({
                    'Annotated total': [sum(errors_counter.values())],
                    'OK': [errors_counter['OK']],
                    'Detection error': [errors_counter['Detection error']],
                    'Classification error': [errors_counter['Classification error']],
                }, index=['count']).T
                df_errors['Percentage'] = df_errors['count'] / total
                df_accuracy = pd.DataFrame({
                    'accuracy': [df_errors.loc['OK', 'count'] / total if total != 0 else 0]
                }, index=['value']).T
                df_errors_placeholder.dataframe(data=df_errors)
                df_accuracy_placeholder.dataframe(data=df_accuracy)
    else:
        if image_data is not None:
            image_data = get_image_data_filtered_by_labels(
                image_data=image_data,
                filter_by_labels=filter_by_labels
            )
            if show_annotation:
                image = cached_visualize_image_data(
                    image_data=image_data,
                    use_labels=use_labels,
                    draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,  # noqa: E501
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
                    draw_rectangle_with_color=[0, 255, 0],
                    show_top_n=show_top_n,
                    average_maximum_images_per_page=average_maximum_images_per_page
                )
            else:
                image = image_data.open_image()
                st.image(image=image, use_column_width=True)
