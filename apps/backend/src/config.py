from yacs.config import CfgNode

cfg = CfgNode()

cfg.system = CfgNode()
cfg.system.use_gpu = False

cfg.models = CfgNode()

object_detection_api = CfgNode()
object_detection_api.description = 'Object Detection API model (from checkpoint)'
object_detection_api.config_path = 'path1'
object_detection_api.checkpoint_path = 'path2'
object_detection_api.score_threshold = 0.3
object_detection_api.model_index = 'detection_model1'

object_detection_api_pb = CfgNode()
object_detection_api_pb.description = 'Object Detection API model (from saved_model.pb)'
object_detection_api_pb.saved_model_dir = 'saved_model_dir/'
object_detection_api_pb.input_type = 'float_image_tensor'  # "image_tensor", "float_image_tensor", "encoded_image_string_tensor"
object_detection_api_pb.score_threshold = 0.3
object_detection_api_pb.model_index = 'detection_model2'

object_detection_api_tflite = CfgNode()
object_detection_api_tflite.description = 'Object Detection API model (from TFLite)'
object_detection_api_tflite.model_path = 'path4'
object_detection_api_tflite.bboxes_output_index = 0
object_detection_api_tflite.scores_output_index = 1
object_detection_api_tflite.score_threshold = 0.3
object_detection_api_tflite.model_index = 'detection_model3'

cfg.models.detection = [
    object_detection_api,
    object_detection_api_pb,
    object_detection_api_tflite
]
# Classification models: 'TensorFlow'
tensorflow_cls_model = CfgNode()
tensorflow_cls_model.description = 'Classficiation Tensorflow Keras Model'
tensorflow_cls_model.input_size = (224, 224)
tensorflow_cls_model.preprocess_input_script_file = './preprocess_input.py'
tensorflow_cls_model.class_names = 'class_names.json'
tensorflow_cls_model.model_path = 'path5'
tensorflow_cls_model.saved_model_type = 'tf.keras'  # 'tf.saved_model', 'tf.keras', tflite'
tensorflow_cls_model.model_index = 'classification_model1'

dummy_cls_model = CfgNode()
dummy_cls_model.description = 'Classficiation Tensorflow Keras Dummy Model'
dummy_cls_model.default_class_name = 'dummy'
dummy_cls_model.model_index = 'classification_model2'
cfg.models.classification = [tensorflow_cls_model, dummy_cls_model]


def get_cfg_defaults():
    return cfg.clone()
