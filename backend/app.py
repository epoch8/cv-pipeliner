from dataclasses import dataclass
from typing import Dict

from flask import Flask, request, jsonify

from src.config import get_cfg_defaults
from src.model import load_pipeline_inferencer, inference, realtime_inference
from src.realtime_inferencer import RealTimeInferencer

app = Flask(__name__)
if 'CV_PIPELINER_BACKEND_MODEL_CONFIG' not in app.config:
    app.config['CV_PIPELINER_BACKEND_MODEL_CONFIG'] = 'config.yaml'
config_file = app.config['CV_PIPELINER_BACKEND_MODEL_CONFIG']
cfg = get_cfg_defaults()
cfg.merge_from_file(config_file)
cfg.freeze()

pipeline_inferencer = load_pipeline_inferencer(cfg)
guid_to_realtime_inferencer_data = {}


@dataclass
class RealTimeInferencerData:
    guid: str
    realtime_inferencer: RealTimeInferencer


@app.route('/realtime_start/<guid>', methods=['POST'])
def realtime_start(guid: str) -> Dict:
    if request.method == 'POST':
        if guid in guid_to_realtime_inferencer_data:
            return jsonify(
                sucess=False,
                message='Realtime process with given guid is already started.'
            ), 400
        else:
            guid_to_realtime_inferencer_data[guid] = RealTimeInferencerData(
                guid=guid,
                realtime_inferencer=RealTimeInferencer(
                    pipeline_inferencer=pipeline_inferencer,
                    fps=float(request.form['fps']),
                    detection_delay=int(request.form['detection_delay'])
                )
            )
            return jsonify(sucess=True)


@app.route('/realtime_predict/<guid>', methods=['POST'])
def realtime_predict(guid: str) -> Dict:
    if request.method == 'POST' and request.files.get('image', '') and guid in guid_to_realtime_inferencer_data:
        res_json = realtime_inference(
            realtime_inferencer=guid_to_realtime_inferencer_data[guid].realtime_inferencer,
            image_bytes=request.files.get('image', ''),
            detection_score_threshold=cfg.models.detection.object_detection_api_tflite.score_threshold,
        )
        return res_json
    return jsonify(success=False, message='Realtime process with given guid is not started.'), 400


@app.route('/realtime_end/<guid>', methods=['POST'])
def realtime_end(guid: str) -> Dict:
    if request.method == 'POST':
        if guid not in guid_to_realtime_inferencer_data:
            return jsonify(sucess=False, message='Realtime process with given guid is not started.'), 400
        else:
            del guid_to_realtime_inferencer_data[guid]
            return jsonify(sucess=True)


@app.route('/predict/', methods=['POST'])
def predict() -> Dict:
    if request.method == 'POST' and request.files.get('image', ''):
        res_json = inference(
            pipeline_inferencer=pipeline_inferencer,
            image_bytes=request.files.get('image', ''),
            detection_score_threshold=cfg.models.detection.object_detection_api_tflite.score_threshold
        )
        return res_json
    return jsonify(success=False)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
