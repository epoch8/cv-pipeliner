import pickle
from pathlib import Path
from typing import Union, List

import pandas as pd
import nbformat as nbf

from two_stage_pipeliner.core.reporter import Reporter
from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer
from two_stage_pipeliner.metrics.pipeline import get_df_pipeline_metrics
from two_stage_pipeliner.visualizers.pipeline import PipelineVisualizer
from two_stage_pipeliner.inference_models.pipeline import PipelineModel
from two_stage_pipeliner.logging import logger

DETECTION_MODEL_SPEC_FILENAME = "detection_model_spec.pkl"
CLASSIFICATION_MODEL_SPEC_FILENAME = "classification_model_spec.pkl"
IMAGES_DATA_FILENAME = "images_data.pkl"


def pipeline_interactive_work(directory: Union[str, Path],
                              detection_score_threshold: float,
                              minimum_iou: float):
    directory = Path(directory)
    detection_model_spec_filepath = directory / DETECTION_MODEL_SPEC_FILENAME
    with open(detection_model_spec_filepath, "rb") as src:
        detection_model_spec = pickle.load(src)
    classification_model_spec_filepath = directory / CLASSIFICATION_MODEL_SPEC_FILENAME
    with open(classification_model_spec_filepath, "rb") as src:
        classification_model_spec = pickle.load(src)
    detection_model = detection_model_spec.load()
    classification_model = classification_model_spec.load()
    classification_model.load(classification_model_spec)
    pipeline_model = PipelineModel()
    pipeline_model.load((detection_model, classification_model))

    pipeline_inferencer = PipelineInferencer(pipeline_model)

    images_data_filepath = directory / IMAGES_DATA_FILENAME
    with open(images_data_filepath, "rb") as src:
        images_data = pickle.load(src)
    pipeline_visualizer = PipelineVisualizer(pipeline_inferencer)
    pipeline_visualizer.visualize(images_data, detection_score_threshold=detection_score_threshold,
                                  show_TP_FP_FN_with_minimum_iou=minimum_iou)


class PipelineReporter(Reporter):
    def _get_markdowns(self,
                       df_pipeline_metrics: pd.DataFrame,
                       df_pipeline_metrics_soft: pd.DataFrame) -> List[str]:
        empty_text = '- To be written.'
        markdowns = []
        markdowns.append(
            '# Task\n'
            '**Input**: Images.\n\n'
            '**Output**: Detect all bounding boxes and classify them.\n'
        )
        markdowns.append(
            '# Pipeline\n'
            '1. **Detection**: the detector predicts with a rectangles (bboxes).\n'
            '2. **Classification**: the classifier makes predictions on the selected bboxes.\n'
        )

        markdowns.append(
            '# Result\n'
            f'{empty_text}''\n'
        )
        markdowns.append(
            '## Common pipeline metrics (strict)\n'
            f'{df_pipeline_metrics.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Common pipeline metrics (soft)\n'
            f'{df_pipeline_metrics_soft.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Interactive work:\n'
        )
        return markdowns

    def _get_codes(self,
                   detection_score_threshold: float,
                   minimum_iou: float) -> List[str]:
        codes = []
        codes.append(f'''
from two_stage_pipeliner.reporters.pipeline import pipeline_interactive_work
pipeline_interactive_work(
    directory='.',
    detection_score_threshold={detection_score_threshold},
    minimum_iou={minimum_iou}
)''')
        codes = [code.strip() for code in codes]
        return codes

    def report(self,
               inferencer: PipelineInferencer,
               true_images_data: List[ImageData],
               directory: Union[str, Path],
               detection_score_threshold: float,
               minimum_iou: float,
               extra_bbox_label: str = None):

        images_data_gen = BatchGeneratorImageData(true_images_data, batch_size=16)
        pred_images_data = inferencer.predict(images_data_gen, detection_score_threshold=detection_score_threshold)
        df_pipeline_metrics = get_df_pipeline_metrics(
            true_images_data=true_images_data,
            pred_images_data=pred_images_data,
            minimum_iou=minimum_iou,
            extra_bbox_label=extra_bbox_label
        )
        df_pipeline_metrics_soft = get_df_pipeline_metrics(
            true_images_data=true_images_data,
            pred_images_data=pred_images_data,
            minimum_iou=minimum_iou,
            extra_bbox_label=extra_bbox_label,
            use_soft_with_known_labels=inferencer.model.classification_model.class_names
        )
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        detection_model_spec_filepath = directory / DETECTION_MODEL_SPEC_FILENAME
        classification_model_spec_filepath = directory / CLASSIFICATION_MODEL_SPEC_FILENAME

        detection_model_spec = inferencer.model.detection_model.model_spec
        classification_model_spec = inferencer.model.classification_model.model_spec
        with open(detection_model_spec_filepath, 'wb') as out:
            pickle.dump(detection_model_spec, out)
        with open(classification_model_spec_filepath, 'wb') as out:
            pickle.dump(classification_model_spec, out)
        images_data_filepath = directory / IMAGES_DATA_FILENAME
        with open(images_data_filepath, 'wb') as out:
            pickle.dump(images_data_gen.data, out)

        markdowns = self._get_markdowns(df_pipeline_metrics, df_pipeline_metrics_soft)
        codes = self._get_codes(
            detection_score_threshold=detection_score_threshold,
            minimum_iou=minimum_iou
        )

        nb = nbf.v4.new_notebook()
        nb['cells'] = [
            nbf.v4.new_markdown_cell(markdown)
            for markdown in markdowns
        ]
        nb['cells'].extend([
            nbf.v4.new_code_cell(code)
            for code in codes
        ])
        nbf.write(nb, str(directory / 'report.ipynb'))
        logger.info(f"Pipeline report saved to '{directory}'.")
