import pickle
from pathlib import Path
from typing import Union, List

import pandas as pd
import nbformat as nbf

from two_stage_pipeliner.core.reporter import Reporter
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer
from two_stage_pipeliner.metrics_counters.pipeline import PipelineMetricsCounter
from two_stage_pipeliner.visualizers.pipeline import PipelineVisualizer
from two_stage_pipeliner.inference_models.detection import checkpoint_to_detection_model
from two_stage_pipeliner.inference_models.classification import checkpoint_to_classification_model

from two_stage_pipeliner.inference_models.pipeline import Pipeline

from two_stage_pipeliner.logging import logger

CHECKPOINT_FILENAME = "checkpoint.pkl"
IMAGES_DATA_FILENAME = "images_data.pkl"


def pipeline_interactive_work(directory: Union[str, Path],
                              detection_score_threshold: float,
                              minimum_iou: float):
    directory = Path(directory)
    checkpoint_filepath = directory / CHECKPOINT_FILENAME
    with open(checkpoint_filepath, "rb") as src:
        detection_checkpoint, classification_checkpoint = pickle.load(src)
    detection_model = checkpoint_to_detection_model(detection_checkpoint)()
    detection_model.load(detection_checkpoint)
    classification_model = checkpoint_to_classification_model(classification_checkpoint)()
    classification_model.load(classification_checkpoint)
    pipeline_model = Pipeline()
    pipeline_model.load((detection_model, classification_model))

    pipeline_inferencer = PipelineInferencer(pipeline_model)

    images_data_filepath = directory / IMAGES_DATA_FILENAME
    with open(images_data_filepath, "rb") as src:
        images_data = pickle.load(src)
    pipeline_visualizer = PipelineVisualizer(pipeline_inferencer)
    pipeline_visualizer.visualize(images_data, detection_score_threshold=detection_score_threshold,
                                  show_TP_FP_FN=True, minimum_iou=minimum_iou)


class PipelineReporter(Reporter):
    def _get_markdowns(self,
                       df_pipeline_metrics: pd.DataFrame) -> List[str]:
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
            '## Common pipeline metrics\n'
            f'{df_pipeline_metrics.to_markdown(stralign="center")}''\n'
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
               data_generator: BatchGeneratorImageData,
               directory: Union[str, Path],
               detection_score_threshold: float,
               minimum_iou: float):

        metrics_counter = PipelineMetricsCounter(inferencer)
        df_pipeline_metrics = metrics_counter.score(
            data_generator, detection_score_threshold, minimum_iou
        )
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        checkpoint_filepath = directory / CHECKPOINT_FILENAME

        detection_checkpoint = inferencer.model.detection_model.checkpoint
        classification_checkpoint = inferencer.model.classification_model.checkpoint
        pipeline_checkpoint = (detection_checkpoint, classification_checkpoint)
        with open(checkpoint_filepath, 'wb') as out:
            pickle.dump(pipeline_checkpoint, out)
        images_data_filepath = directory / IMAGES_DATA_FILENAME
        with open(images_data_filepath, 'wb') as out:
            pickle.dump(data_generator.data, out)

        markdowns = self._get_markdowns(df_pipeline_metrics)
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
        logger.info(f"Detector report saved to '{directory}'.")
