import pickle
from pathlib import Path
from typing import Union, List

import pandas as pd
import nbformat as nbf

from two_stage_pipeliner.core.reporter import Reporter
from two_stage_pipeliner.core.batch_generator import BatchGeneratorImageData
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.metrics_counters.detection import DetectionMetricsCounter
from two_stage_pipeliner.visualizers.detection import DetectionVisualizer
from two_stage_pipeliner.inference_models.detection.checkpoint_to_detection_model import checkpoint_to_detection_model

from two_stage_pipeliner.logging import logger

CHECKPOINT_FILENAME = "checkpoint.pkl"
IMAGES_DATA_FILENAME = "images_data.pkl"


def detection_interactive_work(directory: Union[str, Path],
                               score_threshold: float,
                               minimum_iou: float):
    directory = Path(directory)
    checkpoint_filepath = directory / CHECKPOINT_FILENAME
    with open(checkpoint_filepath, "rb") as src:
        checkpoint = pickle.load(src)
    detection_model = checkpoint_to_detection_model(checkpoint)()
    detection_model.load(checkpoint)

    detection_inferencer = DetectionInferencer(detection_model)

    images_data_filepath = directory / IMAGES_DATA_FILENAME
    with open(images_data_filepath, "rb") as src:
        images_data = pickle.load(src)
    detection_visualizer = DetectionVisualizer(detection_inferencer)
    detection_visualizer.visualize(images_data, score_threshold=score_threshold,
                                   show_TP_FP_FN=True, minimum_iou=minimum_iou)


class DetectionReporter(Reporter):
    def _get_markdowns(self,
                       df_detector_metrics: pd.DataFrame,
                       df_detector_metrics_recall: pd.DataFrame) -> List[str]:
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
            '## Common detector metrics\n'
            f'{df_detector_metrics.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## General recall by class\n'
            f'{df_detector_metrics_recall.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Interactive work:\n'
        )
        return markdowns

    def _get_codes(self,
                   score_threshold: float,
                   minimum_iou: float) -> List[str]:
        codes = []
        codes.append(f'''
from two_stage_pipeliner.reporters.detection import detection_interactive_work
detection_interactive_work(
    directory='.',
    score_threshold={score_threshold},
    minimum_iou={minimum_iou}
)''')
        codes = [code.strip() for code in codes]
        return codes

    def report(self,
               inferencer: DetectionInferencer,
               data_generator: BatchGeneratorImageData,
               directory: Union[str, Path],
               score_threshold: float,
               minimum_iou: float):

        metrics_counter = DetectionMetricsCounter(inferencer)
        df_detector_metrics, df_detector_metrics_recall = metrics_counter.score(
            data_generator, score_threshold, minimum_iou
        )
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        checkpoint_filepath = directory / CHECKPOINT_FILENAME
        with open(checkpoint_filepath, 'wb') as out:
            pickle.dump(inferencer.model.checkpoint, out)
        images_data_filepath = directory / IMAGES_DATA_FILENAME
        with open(images_data_filepath, 'wb') as out:
            pickle.dump(data_generator.data, out)

        markdowns = self._get_markdowns(df_detector_metrics, df_detector_metrics_recall)
        codes = self._get_codes(
            score_threshold=score_threshold,
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
