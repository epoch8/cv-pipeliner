from yacs.config import CfgNode

cfg = CfgNode()

cfg.system = CfgNode()

cfg.data = CfgNode()
cfg.data.base_labels_images_dir = 'renders/'
cfg.data.labels_decriptions = 'label_to_description.json'
cfg.data.images_dirs = [
    {'images_dir_with_annotation/': ['annotations_filename.json']},
    {'images_dir_without_annotation/': []}
]
cfg.data.images_annotation_type = 'supervisely'  # 'supervisely', 'brickit'


def get_cfg_defaults():
    return cfg.clone()
