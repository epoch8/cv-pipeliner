from two_stage_pipeliner.inference_models.detection.tf.specs import DetectorModelSpecTF
from two_stage_pipeliner.inference_models.detection.tf.specs_pb import DetectorModelSpecTF_pb

from two_stage_pipeliner.inference_models.detection.tf.detector import DetectorTF
from two_stage_pipeliner.inference_models.detection.tf.detector_pb import DetectorTF_pb


def checkpoint_to_detection_model(checkpoint):
    if isinstance(checkpoint, DetectorModelSpecTF):
        return DetectorTF
    if isinstance(checkpoint, DetectorModelSpecTF_pb):
        return DetectorTF_pb

    raise ValueError("Unknown checkpoint")
