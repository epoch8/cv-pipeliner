from dataclasses import dataclass
from typing import List, Type
from cv_pipeliner.core.inferencer import Inferencer

from tqdm import tqdm

from cv_pipeliner.core.data import ImageData
from cv_pipeliner.core.inference_model import ModelSpec
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.inferencers.pipeline import PipelineInferencer
from cv_pipeliner.inference_models.pipeline import PipelineModelSpec


@dataclass
class TextDetectionModelSpec(ModelSpec):
    pipeline_model_spec_step1: PipelineModelSpec
    pipeline_model_spec_step2: PipelineModelSpec

    @property
    def inference_model_cls(self) -> Type['TextDetection']:
        from cv_pipeliner.complex_pipelines.text_detection import TextDetection
        return TextDetection


class TextDetection(Inferencer):
    def __init__(
        self,
        model_spec: TextDetectionModelSpec
    ):
        self.pipeline_model_step1 = model_spec.pipeline_model_spec_step1.load()
        self.pipeline_model_step2 = model_spec.pipeline_model_spec_step2.load()

        self.pipeline_inferencer_step1 = PipelineInferencer(self.pipeline_model_step1)
        self.pipeline_inferencer_step2 = PipelineInferencer(self.pipeline_model_step2)

    def predict(
        self,
        images_data_gen: BatchGeneratorImageData,
        score_threshold_step1: float,
        score_threshold_step2: float,
        x_offset: float,
        y_offset: float,
        disable_tqdm: bool = False,
    ) -> List[ImageData]:

        pred_images_data = self.pipeline_inferencer_step1.predict(
            images_data_gen=images_data_gen,
            detection_score_threshold=score_threshold_step1,
            disable_tqdm=True,
            open_images_in_images_data=True
        )

        for pred_image_data in tqdm(pred_images_data, disable=disable_tqdm):
            bboxes_as_images_data = []
            image = pred_image_data.open_image()
            for bbox_data in pred_image_data.bboxes_data:
                cropped_image = bbox_data.open_cropped_image(source_image=image)
                height, width, _ = cropped_image.shape
                x_offset_pixels = int(x_offset * width)
                y_offset_pixels = int(y_offset * height)
                bboxes_as_images_data.append(ImageData(
                    image=bbox_data.open_cropped_image(
                        source_image=image,
                        xmin_offset=x_offset_pixels,
                        ymin_offset=y_offset_pixels,
                        xmax_offset=x_offset_pixels,
                        ymax_offset=y_offset_pixels,
                    )
                ))

            bboxes_as_images_data_gen = BatchGeneratorImageData(
                bboxes_as_images_data,
                batch_size=images_data_gen.batch_size,
                use_not_caught_elements_as_last_batch=True
            )
            pred_images_data_step2 = self.pipeline_inferencer_step2.predict(
                bboxes_as_images_data_gen,
                detection_score_threshold=score_threshold_step2,
                disable_tqdm=True
            )

            for bbox_as_image_data, bbox_data, pred_image_data_step2 in zip(
                bboxes_as_images_data, pred_image_data.bboxes_data, pred_images_data_step2
            ):
                height, width, _ = bbox_as_image_data.image.shape
                if height > width:
                    key = lambda bbox_data: bbox_data.ymin  # noqa: E731
                else:
                    key = lambda bbox_data: bbox_data.xmin  # noqa: E731
                # Sort characters
                bboxes_data_step2_sorted = sorted(pred_image_data_step2.bboxes_data, key=key)
                result_text = ''.join([bbox_data.label for bbox_data in bboxes_data_step2_sorted])
                bbox_data.additional_info['result_text'] = result_text

        return pred_images_data
