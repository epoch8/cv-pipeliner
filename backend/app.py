from typing import Dict

from flask import Flask, request

from src.config import get_cfg_defaults
from src.model import load_pipeline_inferencer, inference

app = Flask(__name__)
if 'CV_PIPELINER_BACKEND_MODEL_CONFIG' not in app.config:
    app.config['CV_PIPELINER_BACKEND_MODEL_CONFIG'] = 'config.yaml'
config_file = app.config['CV_PIPELINER_BACKEND_MODEL_CONFIG']
cfg = get_cfg_defaults()
cfg.merge_from_file(config_file)
cfg.freeze()

pipeline_inferencer = load_pipeline_inferencer(cfg)


@app.route('/predict/<int:guid>', methods=['POST', 'GET'])
def predict(guid: int) -> Dict:
    if request.method == 'POST' and request.content_type == 'image/jpg':
        pred_image_data = inference(
            pipeline_inferencer=pipeline_inferencer,
            image_bytes=request.data,
            detection_score_threshold=cfg.models.detection.object_detection_api_tflite.score_threshold
        )
        res_json = {
            'guid': guid,
            'bboxes': [
                {
                    'xmin': int(bbox_data.xmin),
                    'ymin': int(bbox_data.ymin),
                    'xmax': int(bbox_data.xmax),
                    'ymax': int(bbox_data.ymax),
                    'label': bbox_data.label
                }
                for bbox_data in pred_image_data.bboxes_data
            ]
        }
        return res_json

    if request.method == 'GET':
        return "Please, POST the image/jpg!"
