import imageio

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.inference_models.detection.object_detection_api import (
    ObjectDetectionAPI_TFLite_ModelSpec
)
from cv_pipeliner.inference_models.classification.tensorflow import TensorFlow_ClassificationModelSpec
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec
from cv_pipeliner.inferencers.pipeline import PipelineInferencer

from .config import cfg as _cfg


def load_pipeline_inferencer(cfg: _cfg) -> PipelineInferencer:
    detection_cfg = cfg.models.detection.object_detection_api_tflite
    classification_cfg = cfg.models.classification.tensorflow_cls_model
    pipeline_model_spec = PipelineModelSpec(
        detection_model_spec=ObjectDetectionAPI_TFLite_ModelSpec(
            model_path=detection_cfg.model_path,
            bboxes_output_index=detection_cfg.bboxes_output_index,
            scores_output_index=detection_cfg.scores_output_index
        ),
        classification_model_spec=TensorFlow_ClassificationModelSpec(
            input_size=classification_cfg.input_size,
            preprocess_input=classification_cfg.preprocess_input_script_file,
            class_names=classification_cfg.class_names,
            model_path=classification_cfg.model_path,
            saved_model_type=classification_cfg.saved_model_type
        )
    )
    pipeline_model = pipeline_model_spec.load()
    pipeline_inferencer = PipelineInferencer(pipeline_model)
    return pipeline_inferencer


def inference(
    pipeline_inferencer: PipelineInferencer,
    image_bytes: bytes,
    detection_score_threshold: float
) -> ImageData:
    image = imageio.imread(image_bytes, pilmode='RGB')
    image_data = ImageData(
        image=image
    )
    image_data_gen = BatchGeneratorImageData([image_data], batch_size=1,
                                             use_not_caught_elements_as_last_batch=True)
    pred_image_data = pipeline_inferencer.predict(
        image_data_gen,
        detection_score_threshold=detection_score_threshold,
        open_images_in_images_data=False,
        open_cropped_images_in_bboxes_data=False
    )[0]
    return pred_image_data
