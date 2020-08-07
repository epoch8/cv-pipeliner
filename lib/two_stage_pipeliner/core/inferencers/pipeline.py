import abc

from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.core.inference_model import InferenceModel
from two_stage_pipeliner.core.inferencer import Inferencer
from two_stage_pipeliner.core.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.core.inferencers.classification import ClassificationInferencer


class PipelineModel(InferenceModel):
    def __init__(self,
                 detection_inferencer: DetectionInferencer,
                 classification_inferencer: ClassificationInferencer):
        self.detection_inferencer = detection_inferencer
        self.classification_inferencer = classification_inferencer
        super(InferenceModel, self).__init__()


class PipelineInferencer(Inferencer):
    def __init__(self, model: PipelineModel):
        assert isinstance(model, PipelineModel)
        super(PipelineInferencer, self).__init__(model)

    def predict(self, data_generator: BatchGeneratorImageData) -> BatchGeneratorImageData:
        detection_result = self.detection_inferencer.predict(data_generator)
        classification_result = self.classification_inferencer.predict(detection_result)
        pipeline_images_data = []
        for detection_batch, classification_batch in zip(detection_result, classification_result):
            for detection_image_data, classification_bboxes_data in zip(detection_batch, classification_batch):
                pipeline_images_data.append(ImageData(
                    image_path=detection_image_data.image_path,
                    image=detection_image_data.image_path,
                    bboxes_data=classification_bboxes_data
                ))
