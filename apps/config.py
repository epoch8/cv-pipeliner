from pathlib import Path
from typing import Dict, List, Union

import fsspec
from yacs.config import CfgNode

cfg = CfgNode()

cfg.backend = CfgNode()

cfg.backend.system = CfgNode()
cfg.backend.system.use_gpu = False

cfg.backend.models = CfgNode()

object_detection_api = CfgNode()
object_detection_api.description = 'Object Detection API model (from checkpoint)'
object_detection_api.config_path = 'path1'
object_detection_api.checkpoint_path = 'path2'
object_detection_api.score_threshold = 0.3
object_detection_api.model_index = 'model_index1'
object_detection_api.class_names = None

object_detection_api_pb = CfgNode()
object_detection_api_pb.description = 'Object Detection API model (from saved_model.pb)'
object_detection_api_pb.saved_model_dir = 'saved_model_dir/'
object_detection_api_pb.input_type = 'float_image_tensor'  # "image_tensor", "float_image_tensor", "encoded_image_string_tensor"
object_detection_api_pb.score_threshold = 0.3
object_detection_api_pb.model_index = 'model_index2'
object_detection_api_pb.class_names = None

object_detection_api_tflite = CfgNode()
object_detection_api_tflite.description = 'Object Detection API model (from TFLite)'
object_detection_api_tflite.model_path = 'path4'
object_detection_api_tflite.bboxes_output_index = 0
object_detection_api_tflite.scores_output_index = 1
object_detection_api_tflite.score_threshold = 0.3
object_detection_api_tflite.model_index = 'model_index3'
object_detection_api_tflite.class_names = None

object_detection_api_kfserving = CfgNode()
object_detection_api_kfserving.description = 'Object Detection API model (from KFServing)'
object_detection_api_kfserving.url = 'url:predict'
object_detection_api_kfserving.input_name = 'input_name'
object_detection_api_kfserving.score_threshold = 0.3
object_detection_api_kfserving.model_index = 'model_index4'
object_detection_api_kfserving.class_names = None

cfg.backend.models.detection = [
    object_detection_api,
    object_detection_api_pb,
    object_detection_api_tflite,
    object_detection_api_kfserving
]
# Classification models: 'TensorFlow'
tensorflow_cls_model = CfgNode()
tensorflow_cls_model.description = 'Classficiation Tensorflow Keras Model'
tensorflow_cls_model.input_size = (224, 224)
tensorflow_cls_model.preprocess_input_script_file = './preprocess_input.py'
tensorflow_cls_model.class_names = 'class_names.json'
tensorflow_cls_model.model_path = 'path5'
tensorflow_cls_model.saved_model_type = 'tf.keras'  # 'tf.saved_model', 'tf.keras', tflite'
tensorflow_cls_model.model_index = 'model_index5'

# Classification models: 'TensorFlow'
tensorflow_cls_model_kfserving = CfgNode()
tensorflow_cls_model_kfserving.description = 'Classficiation Tensorflow Model (KFServing)'
tensorflow_cls_model_kfserving.url = 'url:predict'
tensorflow_cls_model_kfserving.input_name = 'input_name'
tensorflow_cls_model_kfserving.input_size = (224, 224)
tensorflow_cls_model_kfserving.preprocess_input_script_file = './preprocess_input.py'
tensorflow_cls_model_kfserving.class_names = 'class_names.json'
tensorflow_cls_model_kfserving.model_index = 'model_index6'

dummy_cls_model = CfgNode()
dummy_cls_model.description = 'Classficiation Tensorflow Keras Dummy Model'
dummy_cls_model.default_class_name = 'dummy'
dummy_cls_model.model_index = 'model_index7'
cfg.backend.models.classification = [tensorflow_cls_model, dummy_cls_model]

# Pipeline models (from indexes)
pipeline_model = CfgNode()
pipeline_model.description = 'Pipeline Model (from above)'
pipeline_model.detection_model_index = 'model_index1'
pipeline_model.classification_model_index = 'model_index6'

cfg.backend.models.pipeline = [pipeline_model]

cfg.data = CfgNode()
cfg.data.base_labels_images = 'renders/'
cfg.data.labels_decriptions = None  # 'label_to_description.json'
cfg.data.ann_class_names = None  # 'ann_class_names.json'
cfg.data.label_to_category = None  # 'label_to_category.json'
cfg.data.images_dirs = [
    {'images_dir_with_annotation/': ['annotations_filename.json']},
    {'images_dir_without_annotation/': []}
]
cfg.data.images_annotation_type = 'supervisely'  # 'supervisely', 'brickit'
cfg.data.minimum_iou = 0.5


def get_cfg_defaults():
    return cfg.clone()


def merge_cfg_from_file_fsspec(
    cfg: CfgNode,
    cfg_filename: Union[str, Path],
) -> CfgNode:
    with fsspec.open(cfg_filename, "r") as src:
        load_cfg = cfg.load_cfg(src.read())
    cfg.merge_from_other_cfg(load_cfg)


def merge_cfg_from_string(
    cfg: CfgNode,
    cfg_str: str,
) -> CfgNode:
    load_cfg = cfg.load_cfg(cfg_str)
    cfg.merge_from_other_cfg(load_cfg)


def get_list_cfg_from_dict(d: Dict):
    return [item for sublist in [(str(k), str(v)) for k, v in d.items()] for item in sublist]


def get_cfg_from_dict(d: Dict, possible_cfgs: List[CfgNode]):
    assert len(d) == 1
    key = list(d)[0]
    cfg = None
    for possible_cfg in possible_cfgs:
        if set(dict(d[key])) == set(dict(possible_cfg)):
            cfg = possible_cfg.clone()
            cfg.merge_from_list(get_list_cfg_from_dict(d[key]))

    if cfg is None:
        raise ValueError(f'Got unknown config: {d}')
    return cfg, key
