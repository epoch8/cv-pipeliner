import os
import tempfile
from pathlib import Path
from typing import Union, List
from io import BytesIO

import tensorflow as tf
import numpy as np

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.visualizers.core.image_data import visualize_image_data
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.tracking.video_inferencer import VideoInferencer
from cv_pipeliner.utils.images_datas import get_image_data_filtered_by_labels
from cv_pipeliner.utils.images import get_label_to_base_label_image
from cv_pipeliner.inferencers.pipeline import PipelineInferencer
from cv_pipeliner.inference_models.pipeline import PipelineModel
from src.data import get_images_data_from_dir, get_videos_data_from_dir
from src.model import (
    load_detection_model,
    load_classification_model,
    get_description_to_detection_model_definition_from_config,
    get_description_to_classification_model_definition_from_config
)
from src.config import get_cfg_defaults
from src.visualization import illustrate_bboxes_data

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

if cfg.system.use_gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

label_to_base_label_image = get_label_to_base_label_image(cfg.data.base_labels_images_dir)
description_to_detection_model_definition = get_description_to_detection_model_definition_from_config(cfg)
description_to_classiticaion_model_definition = get_description_to_classification_model_definition_from_config(cfg)

st.sidebar.title("Pipeline")
st.sidebar.header('Detection')
detection_model_description = st.sidebar.selectbox(
    label='Model:',
    options=[None] + [description for description in description_to_detection_model_definition]
)
st.sidebar.header('Classification')
classification_model_description = st.sidebar.selectbox(
    label='Model:',
    options=[None] + [description for description in description_to_classiticaion_model_definition]
)

if detection_model_description is not None:
    detection_model_definition = description_to_detection_model_definition[detection_model_description]
    detection_model = load_detection_model(
        detection_model_spec=detection_model_definition.model_spec,
    )
    st.sidebar.subheader('Detection score threshold')
    detection_score_threshold = st.sidebar.slider(
        label='Threshold',
        min_value=0.,
        max_value=1.,
        value=detection_model_definition.score_threshold,
        step=0.05
    )

if classification_model_description is not None:
    classification_model_definition = description_to_classiticaion_model_definition[classification_model_description]
    classification_model = load_classification_model(
        classification_model_spec=classification_model_definition.model_spec,
    )
    filter_by_labels = st.sidebar.multiselect(
        label="Classes to find",
        options=list(classification_model.class_names),
        default=[]
    )

if detection_model_description is not None and classification_model_description is not None:
    pipeline_model = PipelineModel()
    pipeline_model.load_from_loaded_models(
        detection_model=detection_model,
        classification_model=classification_model
    )
    pipeline_inferencer = PipelineInferencer(pipeline_model)
    run = st.sidebar.checkbox('RUN')
else:
    pipeline_inferencer = None
    filter_by_labels = None
    run = False

st.sidebar.title("Input")
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

input_type = st.sidebar.radio(
    label='Input',
    options=["Image", "Video"]
)

if input_type == 'Image':
    images_from = st.sidebar.selectbox(
        'Image from',
        options=['Upload'] + cfg.data.images_dirs
    )
    if images_from == 'Upload':
        st.header('Image')
        image_bytes = st.file_uploader("Upload image", type=["png", "jpeg", "jpg"])
        if image_bytes is not None:
            image_data = ImageData(image_bytes=image_bytes.getvalue())
        else:
            image_data = None
        show_annotation = False
    else:
        images_data, annotation_success = get_images_data_from_dir(
            images_annotation_type=cfg.data.images_annotation_type,
            images_dir=images_from
        )
        if annotation_success:
            images_data_captions = [
                f"[{i}] {image_data.image_path.name} [{len(image_data.bboxes_data)} bboxes]"
                for i, image_data in enumerate(images_data)
            ]
        else:
            images_data_captions = [
                f"[{i}] {image_data.image_path.name}"
                for i, image_data in enumerate(images_data)
            ]
        images_data_selected_caption = st.sidebar.selectbox(
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
            show_annotation = st.sidebar.checkbox('Show annotation', value=False)
        else:
            show_annotation = False

    mode = st.sidebar.radio(
        label='Output bboxes',
        options=["many", "one-by-one"]
    )

elif input_type == 'Video':
    videos_from = st.sidebar.selectbox(
        'Video from',
        options=['Upload'] + cfg.data.videos_dirs
    )
    st.header('Video')
    if videos_from == 'Upload':
        st.text('Please do not upload large files, they will take longer time to process.')
        video_file = st.file_uploader("Upload video", type=["mp4", "m4v", "mov"])
    else:
        video_paths = get_videos_data_from_dir(videos_dir=videos_from)
        videos_paths_captions = [
            f"[{i}] {video_path.name}"
            for i, video_path in enumerate(video_paths)
        ]
        video_data_selected_caption = st.sidebar.selectbox(
            label='Video',
            options=[None] + videos_paths_captions
        )

        if video_data_selected_caption is not None:
            video_path_caption_index = videos_paths_captions.index(video_data_selected_caption)
            video_path = video_paths[video_path_caption_index]
            with open(video_path, 'rb') as src:
                video_file = BytesIO(src.read())
            st.text(video_data_selected_caption)
        else:
            video_file = None

    if pipeline_inferencer is not None:
        st.sidebar.title("Delays")
        detection_delay = st.sidebar.slider(
            "Detection delay in ms", min_value=0, max_value=500, value=300
        )
        classification_delay = st.sidebar.slider(
            "Classification delay in ms", min_value=0, max_value=500, value=50
        )


@st.cache(show_spinner=False, allow_output_mutation=True)
def inference_one_image(image_data: ImageData,
                        detection_score_threshold: float) -> ImageData:
    image_data_gen = BatchGeneratorImageData([image_data], batch_size=1,
                                             use_not_caught_elements_as_last_batch=True)
    pred_image_data = pipeline_inferencer.predict(
        image_data_gen,
        detection_score_threshold=detection_score_threshold,
        open_images_in_images_data=False,
        open_cropped_images_in_bboxes_data=False
    )[0]
    return pred_image_data


def inference_one_video(
    video_file: Union[str, Path, BytesIO],
    classification_delay: int,
    detection_delay: int,
    detection_score_threshold: float,
    filter_by_labels: List[str],
    draw_base_labels_with_given_label_to_base_label_image: bool
) -> tempfile.NamedTemporaryFile:

    video_inferencer = VideoInferencer(
        pipeline_inferencer=pipeline_inferencer,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
        write_labels=use_labels
    )

    result_video_file = video_inferencer.process_video(
        video_file=video_file,
        classification_delay=classification_delay,
        detection_delay=detection_delay,
        detection_score_threshold=detection_score_threshold,
        filter_by_labels=filter_by_labels
    )

    return result_video_file


@st.cache(show_spinner=False, allow_output_mutation=True)
def cached_visualize_image_data(**kwargs) -> np.ndarray:
    return visualize_image_data(**kwargs)


if input_type == 'Image':
    if run and image_data is not None:
        with st.spinner("Working on your image..."):
            pred_image_data = inference_one_image(
                image_data=image_data,
                detection_score_threshold=detection_score_threshold
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
                mode=mode,
                pred_image_data=pred_image_data,
                minimum_iou=cfg.data.minimum_iou,
                background_color_a=[0, 0, 0, 255],
                true_background_color_b=[0, 255, 0, 255],
                pred_background_color_b=[255, 255, 0, 255]
            )
        else:
            illustrate_bboxes_data(
                true_image_data=pred_image_data,
                label_to_base_label_image=label_to_base_label_image,
                mode=mode,
                background_color_a=[0, 0, 0, 255],
                true_background_color_b=[255, 255, 0, 255]
            )
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
                    draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
                )
                st.image(image=image, use_column_width=True)
                illustrate_bboxes_data(
                    true_image_data=image_data,
                    label_to_base_label_image=label_to_base_label_image,
                    mode=mode,
                    background_color_a=[0, 0, 0, 255],
                    true_background_color_b=[0, 255, 0, 255],
                )
            else:
                image = image_data.open_image()
                st.image(image=image, use_column_width=True)
elif input_type == 'Video':
    if run and video_file is not None and pipeline_inferencer is not None:
        with st.spinner("Working on your video..."):
            result_video_file = inference_one_video(
                video_file=video_file,
                classification_delay=classification_delay,
                detection_delay=detection_delay,
                detection_score_threshold=detection_score_threshold,
                filter_by_labels=filter_by_labels,
                draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image
            )
        st.video(result_video_file, format='video/mp4')
    else:
        if video_file is not None:
            st.video(video_file, format='video/mp4')
