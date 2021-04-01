from dataclasses import asdict
import os
import json
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec
from cv_pipeliner.inferencers.pipeline import PipelineInferencer
from cv_pipeliner.metrics.image_data_matching import ImageDataMatching
from cv_pipeliner.utils.models_definitions import PipelineDefinition
from dacite import from_dict
import fsspec
from typing import Dict, List
from collections import Counter

import numpy as np
from pathy import Pathy

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.visualizers.core.image_data import visualize_image_data
from cv_pipeliner.utils.images_datas import get_image_data_filtered_by_labels
from cv_pipeliner.utils.images import (
    get_image_b64, get_label_to_base_label_image
)
from cv_pipeliner.utils.data import get_label_to_description
from cv_pipeliner.utils.dash.data import get_images_data_from_dir
from apps.config import get_cfg_defaults, merge_cfg_from_string
from apps.model import (
    get_detection_models_definitions_from_config,
    get_classification_models_definitions_from_config,
    inference
)

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from cv_pipeliner.utils.dash.visualization import illustrate_bboxes_data

config_file = os.environ['CV_PIPELINER_APP_CONFIG']
cfg, current_config_str = None, None
label_to_base_label_image, label_to_description, label_to_category = None, None, None
ann_class_names = None
detection_models_definitions, classification_models_definitions = [], []
minimum_iou = None
top_n = 20


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


def read_config_file() -> bool:
    global cfg, current_config_str
    global label_to_base_label_image, label_to_description, label_to_category
    global ann_class_names
    global detection_models_definitions, classification_models_definitions
    global minimum_iou
    with fsspec.open(config_file, 'r') as src:
        config_str = src.read()
    if config_str != current_config_str:
        if current_config_str is not None:
            app.logger.info("Loading config...")
        else:
            app.logger.info("Config change detected. Reloading...")
        cfg = get_cfg_defaults()
        merge_cfg_from_string(cfg, config_str)
        current_config_str = config_str

        label_to_base_label_image = get_label_to_base_label_image(
            base_labels_images=cfg.data.base_labels_images
        )
        label_to_description = get_label_to_description(label_to_description_dict=cfg.data.labels_decriptions)
        label_to_category = get_label_to_description(
            label_to_description_dict=cfg.data.label_to_category,
            default_description='No category'
        )

        if cfg.data.ann_class_names is not None:
            with fsspec.open(cfg.data.ann_class_names, 'r') as src:
                ann_class_names = json.load(src)

        detection_models_definitions = [
            detection_model_definition
            for detection_model_definition in get_detection_models_definitions_from_config(cfg)
        ]
        classification_models_definitions = [
            classification_model_definition
            for classification_model_definition in get_classification_models_definitions_from_config(cfg)
        ]

        minimum_iou = cfg.data.minimum_iou

        return True
    else:
        return False


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "45rem",
    "margin-right": "4rem",
    "padding": "2rem 1rem",
}


sidebar = html.Div(
    children=[
        html.P(
            "Detection model"
        ),
        dcc.Dropdown(
            id='detection_model_definition',
            options=[
                {'label': 'None', 'value': 'None'},
            ],
            optionHeight=120,
        ),
        html.P(
            "Classification model"
        ),
        dcc.Dropdown(
            id='classification_model_definition',
            options=[
                {'label': 'None', 'value': 'None'},
            ],
            optionHeight=120
        ),
        html.Hr(),
        html.P(
            "Images from"
        ),
        dcc.Dropdown(
            id='images_from',
            options=[
                {'label': 'None', 'value': 'None'},
            ],
        ),
        html.P(
            "Annotation filepath"
        ),
        dcc.Dropdown(
            id='annotation_filepath',
            options=[
                {'label': 'None', 'value': 'None'},
            ],
        ),
        html.Hr(),
        html.P(
            "Maximum images per page"
        ),
        html.Center(
            id='average_maximum_images_per_page_output',
            children=[]
        ),
        dcc.Slider(
            id='average_maximum_images_per_page',
            min=1,
            max=100,
            step=1,
            value=20,
            marks={i: {'label': str(i)} for i in range(10, 101, 10)}
        ),
        dcc.Checklist(
            id='use_labels',
            options=[
                {'label': 'Write labels on image', 'value': 'true'},
            ],
            value=['true']
        ),
        dcc.Checklist(
            id='draw_label_images',
            options=[
                {'label': 'Draw base labels images', 'value': 'true'},
            ],
            value=[]
        ),
        dcc.Checklist(
            id='show_annotation',
            options=[
                {'label': 'Show annotation', 'value': 'true'},
            ],
            value=['true']
        ),
        dcc.Checklist(
            id='show_top_n',
            options=[
                {'label': 'Show top-20', 'value': 'true'},
            ],
            value=[]
        ),
        html.Hr(),
        html.Center(
            children=[
                html.P(
                    children='Page 1/1',
                    id='current_page_text'
                )
            ],
            style={
                'font-size': '30px'
            }
        ),
        html.Center(
            children=[
                html.Button(
                    children='Back',
                    id='back_button',
                    style={
                        'width': '100px'
                    }
                ),
                html.Button(
                    children='Next',
                    id='next_button',
                    style={
                        'width': '100px'
                    }
                )
            ],
        )
    ],
    style=SIDEBAR_STYLE
)

stores = html.Div(
    [
        dcc.Interval(
            id='interval-component',
            interval=3*1000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Store(id='config', data=current_config_str),
        dcc.Store(id='pipeline_model_definition', data=None),
        dcc.Store(id='annotation_success', data=False),
        dcc.Store(id='images_data', storage_type='session'),
        dcc.Store(id='current_ann_class_names', storage_type='session'),
        dcc.Store(id='filter_by_labels', storage_type='session'),
        dcc.Store(id='current_image_data', storage_type='session'),
        dcc.Store(id='current_image_data_filtered', storage_type='session'),
        dcc.Store(id='current_pred_image_data', storage_type='session'),
        dcc.Store(id='current_pred_image_data_filtered', storage_type='session'),
        dcc.Store(id='current_page', storage_type='session', data=1),
        dcc.Store(id='maximum_page', storage_type='session', data=1),
    ]
)

main_page_content = html.Div(
    children=[
        html.Div(
            children=[
                html.P(
                    "Classes to find"
                ),
                dcc.Dropdown(
                    id='find_labels',
                    options=[
                        {'label': 'None', 'value': 'None'},
                    ],
                    multi=True
                ),
                html.Br(),
                html.P(
                    "Classes to hide"
                ),
                dcc.Dropdown(
                    id='hide_labels',
                    options=[
                        {'label': 'None', 'value': 'None'},
                    ],
                    multi=True
                ),
                html.Hr(),
                html.Center(
                    html.Button(
                        children='Predict',
                        id='predict_button',
                        style={
                            'height': '50px',
                            'width': '120px'
                        }
                    )
                ),
                html.Br(),
                html.Div(
                    id='images_data_selected_caption_view',
                    children=[
                        html.P(
                            "Choose an image:"
                        ),
                        dcc.Dropdown(
                            id='images_data_selected_caption',
                            options=[
                                {'label': 'None', 'value': 'None'},
                            ]
                        ),
                    ],
                    style={'display': 'none'}
                ),
                dcc.Upload(
                    id='upload_image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select File')
                    ]),
                    style={'display': 'none'}
                ),
                html.Br(),
                html.Div(
                    id='page_content_image',
                    children=[]
                ),
                html.Div(
                    id='page_content_bboxes',
                    children=[]
                )
            ],
            style={
                "max-width": "600px",
            }
        )
    ],
    id="main_page_content",
    style=CONTENT_STYLE
)


app.layout = html.Div([
    stores,
    dcc.Location(id="url"),
    sidebar,
    main_page_content
])


@app.callback(
    Output('config', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_config(n):
    read_config_file()
    return current_config_str


@app.callback(
    dash.dependencies.Output('average_maximum_images_per_page_output', 'children'),
    [dash.dependencies.Input('average_maximum_images_per_page', 'value')])
def update_dcc_slider_output(value):
    return [
        html.P(f"{value}")
    ]


@app.callback(
    Output("current_page_text", "children"),
    [
        Input("current_page", "data"),
        Input("maximum_page", "data")
    ]
)
def render_current_page_text(
    current_page,
    maximum_page,
):
    return f"Page {current_page}/{maximum_page}"


@app.callback(
    Output("current_page", "data"),
    [
        Input("current_page", "data"),
        Input("maximum_page", "data"),
        Input("back_button", "n_clicks"),
        Input("next_button", "n_clicks")
    ]
)
def on_click_back_button(
    current_page,
    maximum_page,
    back_button,
    next_button
):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "back_button" in changed_id:
        current_page -= 1
    elif "next_button" in changed_id:
        current_page += 1

    if current_page > maximum_page:
        current_page = 1
    elif current_page < 1:
        current_page = maximum_page

    return current_page


@app.callback(
    Output("detection_model_definition", "options"),
    [
        Input("config", "data")
    ]
)
def render_detection_models_definitions(data):
    global detection_models_definitions
    dropdown_options = [
        {'label': detection_model_definition.description, 'value': idx}
        for idx, detection_model_definition in enumerate(detection_models_definitions)
    ]
    return dropdown_options


@app.callback(
    Output("classification_model_definition", "options"),
    [
        Input("config", "data")
    ]
)
def render_classification_models_definitions(data):
    global classification_models_definitions
    dropdown_options = [
        {'label': classification_model_definition.description, 'value': idx}
        for idx, classification_model_definition in enumerate(classification_models_definitions)
    ]
    return dropdown_options


@app.callback(
    Output("pipeline_model_definition", "data"),
    [
        Input("detection_model_definition", "value"),
        Input("classification_model_definition", "value")
    ]
)
def get_pipeline_model_spec(
    detection_model_definition_idx,
    classification_model_definition_idx
):
    global detection_models_definitions, classification_models_definitions
    if detection_model_definition_idx is None or classification_model_definition_idx is None:
        return None
    detection_model_definition = detection_models_definitions[detection_model_definition_idx]
    classification_model_definition = classification_models_definitions[classification_model_definition_idx]
    pipeline_model_definition = PipelineDefinition(
        detection_model_definition=detection_model_definition,
        classification_model_definition=classification_model_definition
    )
    return asdict(pipeline_model_definition)


@app.callback(
    Output("images_from", "options"),
    [
        Input("config", "data")
    ]
)
def render_images_dirs(data):
    images_dirs = [list(d)[0] for d in cfg.data.images_dirs]
    image_dir_to_annotation_filepaths = {
        image_dir: d[image_dir] for d, image_dir in zip(cfg.data.images_dirs, images_dirs)
    }
    images_dirs = [image_dir for image_dir in images_dirs if len(image_dir_to_annotation_filepaths[image_dir]) > 0]
    dropdown_options = [{'label': f"../{Pathy(image_dir).name}", 'value': image_dir} for image_dir in images_dirs]
    return dropdown_options


@app.callback(
    Output("annotation_filepath", "options"),
    [
        Input("images_from", "value")
    ]
)
def render_annotation_paths(images_from):
    if images_from is not None:
        images_dirs = [list(d)[0] for d in cfg.data.images_dirs]
        image_dir_to_annotation_filepaths = {
            image_dir: d[image_dir] for d, image_dir in zip(cfg.data.images_dirs, images_dirs)
        }
        dropdown_options = [
            {
                'label': f"../{Pathy(filepath).name}",
                'value': filepath
            } for filepath in image_dir_to_annotation_filepaths[images_from]
        ]
    else:
        dropdown_options = [{'label': 'None', 'value': 'None'}]

    return dropdown_options


@app.callback(
    [
        Output("images_data_selected_caption_view", "style"),
        Output("upload_image", "style"),
    ],
    [
        Input("images_from", "value"),
    ]
)
def render_images_data_selected_caption(
    images_from: str
):
    if images_from is None:
        return {'display': 'none'}, {
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }
    else:
        return {}, {'display': 'none'}


@app.callback(
    [
        Output("images_data", "data"),
        Output("images_data_selected_caption", "options"),
        Output("annotation_success", "data")
    ],
    [
        Input("images_from", "value"),
        Input("annotation_filepath", "value"),
        Input("upload_image", "contents")
    ]
)
def get_images_data(
    images_from: str,
    annotation_filepath: str,
    upload_image_contents: str
):
    images_data = None
    images_data_options = [{'label': 'None', 'value': 'None'}]
    annotation_success = False
    if images_from is not None:
        images_data, annotation_success = get_images_data_from_dir(
            images_annotation_type=cfg.data.images_annotation_type,
            images_dir=images_from,
            annotation_filepath=annotation_filepath,
        )
        images_data = sorted(images_data, key=lambda image_data: image_data.image_name)
        idxs_images_data = [(idx, image_data) for idx, image_data in enumerate(images_data)]
        if annotation_success:
            idxs_images_data = sorted(
                idxs_images_data,
                key=lambda pair: len(pair[1].bboxes_data),
                reverse=True
            )
            images_data_captions = [
                f"[{idx}] {image_data.image_name} [{len(image_data.bboxes_data)} bboxes]"
                for idx, image_data in idxs_images_data
            ]
        else:
            images_data_captions = [
                f"[{idx}] {image_data.image_name}"
                for idx, image_data in idxs_images_data
            ]
        images_data_options = [
            {
                'label': image_data_caption,
                'value': idx
            } for (idx, _), image_data_caption in zip(idxs_images_data, images_data_captions)
        ]
        images_data = [image_data.asdict() for image_data in images_data]
    elif images_from is None and upload_image_contents is not None:
        content_type, content_string = upload_image_contents.split(',')
        images_data = [ImageData(image=content_string).asdict()]
        images_data_options = [{'label': 'Upload', 'value': 0}]

    return images_data, images_data_options, annotation_success


@app.callback(
    [
        Output("current_image_data", "data"),
        Output("current_pred_image_data", "data"),
        Output("current_ann_class_names", "data"),
    ],
    [
        Input("images_data", "data"),
        Input("images_data_selected_caption", "value"),
        Input("images_data_selected_caption", "options"),
        Input("pipeline_model_definition", "data"),
        Input('predict_button', 'n_clicks'),
    ]
)
def update_current_image_data(
    images_data,
    images_data_selected_caption,
    images_data_selected_caption_options,
    pipeline_model_definition,
    predict_button
):
    global ann_class_names
    current_ann_class_names = ann_class_names

    if images_data is None:
        return None, None, current_ann_class_names

    if images_data_selected_caption is None:
        if len(images_data_selected_caption_options) == 1 and (
            images_data_selected_caption_options[0]['label'] == 'Upload'
        ):
            images_data_selected_caption = 0
        else:
            return None, None, current_ann_class_names

    current_image_data = images_data[images_data_selected_caption]
    current_image_data = ImageData.from_dict(current_image_data)
    bboxes_data = current_image_data.bboxes_data

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "predict_button" in changed_id and pipeline_model_definition is not None:
        pipeline_model_definition = from_dict(
            data_class=PipelineDefinition,
            data=pipeline_model_definition
        )
        pipeline_model_spec = PipelineModelSpec(
            detection_model_spec=pipeline_model_definition.detection_model_definition.model_spec,
            classification_model_spec=pipeline_model_definition.classification_model_definition.model_spec
        )
        current_pred_image_data = inference(
            pipeline_model_spec=pipeline_model_spec,
            image_data=current_image_data,
            detection_score_threshold=pipeline_model_definition.detection_model_definition.score_threshold,
            classification_top_n=top_n
        )
        current_pred_image_data.image = current_image_data.image
        bboxes_data = bboxes_data + current_pred_image_data.bboxes_data
    else:
        current_pred_image_data = None

    if current_ann_class_names is None:
        labels = [bbox_data.label for bbox_data in bboxes_data]
        labels_counter = Counter(labels)
        current_ann_class_names = sorted(
            set(labels), key=lambda label: labels_counter[label]
        )

    current_image_data = current_image_data.asdict()
    if current_pred_image_data is not None:
        current_pred_image_data = current_pred_image_data.asdict()

    return current_image_data, current_pred_image_data, current_ann_class_names

@app.callback(
    [
        Output("current_image_data_filtered", "data"),
        Output("current_pred_image_data_filtered", "data"),
        Output("maximum_page", "data"),
    ],
    [
        Input("current_image_data", "data"),
        Input("current_pred_image_data", "data"),
        Input("average_maximum_images_per_page", "value"),
        Input("find_labels", "value"),
        Input("hide_labels", "value"),
    ]
)
def update_current_image_data_filtered_and_maximum_page(
    current_image_data,
    current_pred_image_data,
    average_maximum_images_per_page,
    find_labels,
    hide_labels
):
    if current_image_data is None:
        return None, None, 1

    current_image_data_filtered = ImageData.from_dict(current_image_data)
    current_image_data_filtered = get_image_data_filtered_by_labels(
        image_data=current_image_data_filtered,
        filter_by_labels=find_labels,
        include=True
    )
    current_image_data_filtered = get_image_data_filtered_by_labels(
        image_data=current_image_data_filtered,
        filter_by_labels=hide_labels,
        include=False
    )

    if current_pred_image_data is not None:
        current_pred_image_data_filtered = ImageData.from_dict(current_pred_image_data)
        current_pred_image_data_filtered = get_image_data_filtered_by_labels(
            image_data=current_pred_image_data_filtered,
            filter_by_labels=find_labels,
            include=True
        )
        current_pred_image_data_filtered = get_image_data_filtered_by_labels(
            image_data=current_pred_image_data_filtered,
            filter_by_labels=hide_labels,
            include=False
        )
    else:
        current_pred_image_data_filtered = None

    if current_pred_image_data_filtered is not None:
        global minimum_iou
        image_data_matching = ImageDataMatching(
            true_image_data=current_image_data_filtered,
            pred_image_data=current_pred_image_data_filtered,
            minimum_iou=minimum_iou
        )
        bboxes_data = [
            bbox_data_matching for bbox_data_matching in image_data_matching.bboxes_data_matchings
            if bbox_data_matching.pred_bbox_data is not None
        ]
    else:
        bboxes_data = current_image_data_filtered.bboxes_data

    maximum_page = max(1, int(np.ceil(len(bboxes_data) / average_maximum_images_per_page)))

    current_image_data_filtered = current_image_data_filtered.asdict()
    if current_pred_image_data_filtered is not None:
        current_pred_image_data_filtered = current_pred_image_data_filtered.asdict()

    return current_image_data_filtered, current_pred_image_data_filtered, maximum_page

@app.callback(
    [
        Output("find_labels", "options"),
        Output("hide_labels", "options")
    ],
    [
        Input("current_ann_class_names", "data"),
    ]
)
def update_find_and_hide_labels(
    current_ann_class_names
):
    if current_ann_class_names is None:
        res_options = [{'label': 'None', 'value': 'None'}]
    else:
        res_options = [
            {
                'label': f'{class_name} [{label_to_description[class_name]}]',
                'value': class_name
            }
            for class_name in current_ann_class_names
        ]
    return res_options, res_options


@app.callback(
    Output("page_content_image", "children"),
    [
        Input("current_image_data_filtered", "data"),
        Input("current_pred_image_data_filtered", "data"),
        Input('show_annotation', "value"),
        Input('use_labels', "value"),
        Input('draw_label_images', "value")
    ]
)
def render_main_image(
    current_image_data_filtered,
    current_pred_image_data_filtered,
    show_annotation,
    use_labels,
    draw_label_images
):

    div_children_result = []

    show_annotation = True if 'true' in show_annotation else False
    if current_image_data_filtered is None:
        return None
    if current_pred_image_data_filtered is not None:
        image_data = ImageData.from_dict(current_pred_image_data_filtered)
    else:
        image_data = ImageData.from_dict(current_image_data_filtered)
        if not show_annotation:
            image_data.bboxes_data = []
    use_labels = True if 'true' in use_labels else False
    draw_label_images = True if 'true' in draw_label_images else False
    global label_to_base_label_image
    draw_base_labels_with_given_label_to_base_label_image = (
        label_to_base_label_image if draw_label_images else None
    )

    image = visualize_image_data(
        image_data=image_data,
        use_labels=use_labels,
        draw_base_labels_with_given_label_to_base_label_image=draw_base_labels_with_given_label_to_base_label_image,
    )
    div_children_result = [
        html.Hr(),
        html.Img(
            src=f"data:image/png;base64,{get_image_b64(image, format='png')}",
            style={
                'max-width': '100%',
                'height': 'auto'
            }
        ),
        html.Hr(),
        html.Hr()
    ]

    return div_children_result


@app.callback(
    Output("page_content_bboxes", "children"),
    [
        Input("current_image_data_filtered", "data"),
        Input("current_pred_image_data_filtered", "data"),
        Input('show_annotation', "value"),
        Input("annotation_success", "data"),
        Input('show_top_n', "value"),
        Input("average_maximum_images_per_page", "value"),
        Input("current_page", "data"),
    ]
)
def render_bboxes(
    current_image_data_filtered,
    current_pred_image_data_filtered,
    show_annotation,
    annotation_success,
    show_top_n,
    average_maximum_images_per_page,
    current_page,
):
    if current_image_data_filtered is None:
        return None
    show_annotation = True if 'true' in show_annotation else False
    show_top_n = True if 'true' in show_top_n else False

    true_image_data = ImageData.from_dict(current_image_data_filtered)
    if current_pred_image_data_filtered is not None:
        pred_image_data = ImageData.from_dict(current_pred_image_data_filtered)
    else:
        pred_image_data = None
    if show_annotation and annotation_success:
        return illustrate_bboxes_data(
            true_image_data=true_image_data,
            label_to_base_label_image=label_to_base_label_image,
            label_to_description=label_to_description,
            pred_image_data=pred_image_data,
            minimum_iou=cfg.data.minimum_iou,
            background_color_a=[0, 0, 0, 255],
            true_background_color_b=[0, 255, 0, 255],
            pred_background_color_b=[255, 255, 0, 255],
            bbox_offset=100,
            draw_rectangle_with_color=[0, 255, 0],
            show_top_n=show_top_n,
            average_maximum_images_per_page=average_maximum_images_per_page,
            current_page=current_page
        )
    elif pred_image_data is not None:
        return illustrate_bboxes_data(
            true_image_data=pred_image_data,
            label_to_base_label_image=label_to_base_label_image,
            label_to_description=label_to_description,
            background_color_a=[0, 0, 0, 255],
            true_background_color_b=[255, 255, 0, 255],
            bbox_offset=100,
            draw_rectangle_with_color=[0, 255, 0],
            show_top_n=show_top_n,
            average_maximum_images_per_page=average_maximum_images_per_page,
            current_page=current_page
        )
    else:
        return None


if __name__ == "__main__":
    read_config_file()
    app.run_server(debug=True)
