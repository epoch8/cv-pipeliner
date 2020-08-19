from two_stage_pipeliner.core.inference_model import Checkpoint
from two_stage_pipeliner.inference_models.classification.core import ClassificationModel

from two_stage_pipeliner.inference_models.classification.tf.specs import ClassifierModelSpecTF

from two_stage_pipeliner.inference_models.classification.tf.classifier import ClassifierTF


def load_classification_model_from_checkpoint(checkpoint: Checkpoint) -> ClassificationModel:
    classification_model = None

    if isinstance(checkpoint, ClassifierModelSpecTF):
        classification_model = ClassifierTF()

    if classification_model is not None:
        classification_model.load(checkpoint)
        return classification_model
    else:
        raise ValueError("Unknown checkpoint")
