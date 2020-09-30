from typing import List, Dict

import imageio

from cv_pipeliner.core.data import BboxData, ImageData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.inference_models.detection.object_detection_api import (
    ObjectDetectionAPI_TFLite_ModelSpec
)
from cv_pipeliner.inference_models.classification.tensorflow import TensorFlow_ClassificationModelSpec
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec
from cv_pipeliner.inferencers.pipeline import PipelineInferencer

from src.config import cfg as _cfg
from src.realtime_inferencer import RealTimeInferencer


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


def bboxes_data_to_json(bboxes_data: List[BboxData]) -> Dict:
    json_res = {
        'bboxes': [
            {
                'xmin': int(bbox_data.xmin),
                'ymin': int(bbox_data.ymin),
                'xmax': int(bbox_data.xmax),
                'ymax': int(bbox_data.ymax),
                'label': bbox_data.label,
            }
            for bbox_data in bboxes_data
        ]
    }
    return json_res


def inference(
    pipeline_inferencer: PipelineInferencer,
    image_bytes: bytes,
    detection_score_threshold: float
) -> Dict:
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
    json_res = bboxes_data_to_json(pred_image_data.bboxes_data)

    return json_res


def realtime_inference(
    realtime_inferencer: RealTimeInferencer,
    image_bytes: bytes,
    detection_score_threshold: float,
    batch_size: int = 16
) -> Dict:
    image = imageio.imread(image_bytes, pilmode='RGB')
    pred_bboxes_data = realtime_inferencer.predict_on_frame(
        frame=image,
        detection_score_threshold=detection_score_threshold,
        batch_size=batch_size
    )
    json_res = bboxes_data_to_json(pred_bboxes_data)
    
    return json_res
