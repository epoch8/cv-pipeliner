from yacs.config import CfgNode

cfg = CfgNode()

cfg.models = CfgNode()
cfg.models.detection = CfgNode()

detection = cfg.models.detection
detection.object_detection_api_tflite = CfgNode()
detection.object_detection_api_tflite.description = 'Object Detection API model (from TFLite)'
detection.object_detection_api_tflite.model_path = 'path1'
detection.object_detection_api_tflite.bboxes_output_index = 0
detection.object_detection_api_tflite.scores_output_index = 1
detection.object_detection_api_tflite.score_threshold = 0.3

cfg.models.classification = CfgNode()
classification = cfg.models.classification
classification.tensorflow_cls_model = CfgNode()
classification.tensorflow_cls_model.description = 'Classficiation Tensorflow Keras Model (from TFLite)'
classification.tensorflow_cls_model.input_size = (224, 224)
classification.tensorflow_cls_model.preprocess_input_script_file = './preprocess_input.py'
classification.tensorflow_cls_model.class_names = 'class_names.json'
classification.tensorflow_cls_model.model_path = 'path2'
classification.tensorflow_cls_model.saved_model_type = 'tf.keras'


def get_cfg_defaults():
    return cfg.clone()
