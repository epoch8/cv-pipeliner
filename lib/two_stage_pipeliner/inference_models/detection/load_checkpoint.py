from two_stage_pipeliner.core.inference_model import Checkpoint
from two_stage_pipeliner.inference_models.detection.core import DetectionModel

from two_stage_pipeliner.inference_models.detection.tf.specs import DetectorModelSpecTF
from two_stage_pipeliner.inference_models.detection.tf.specs_pb import DetectorModelSpecTF_pb

from two_stage_pipeliner.inference_models.detection.tf.detector import DetectorTF
from two_stage_pipeliner.inference_models.detection.tf.detector_pb import DetectorTF_pb


def load_detection_model_from_checkpoint(checkpoint: Checkpoint) -> DetectionModel:
    detection_model = None

    if isinstance(checkpoint, DetectorModelSpecTF):
        detection_model = DetectorTF()
    if isinstance(checkpoint, DetectorModelSpecTF_pb):
        detection_model = DetectorTF_pb()

    if detection_model is not None:
        detection_model.load(checkpoint)
        return detection_model
    else:
        raise ValueError("Unknown checkpoint")
