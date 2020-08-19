from two_stage_pipeliner.inference_models.classification.tf.specs import ClassifierModelSpecTF

from two_stage_pipeliner.inference_models.classification.tf.classifier import ClassifierTF


def checkpoint_to_classification_model(checkpoint):
    if isinstance(checkpoint, ClassifierModelSpecTF):
        return ClassifierTF

    raise ValueError("Unknown checkpoint")
