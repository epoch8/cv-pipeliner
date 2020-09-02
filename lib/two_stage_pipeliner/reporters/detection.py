import pickle
from pathlib import Path
from typing import Union, List

import pandas as pd
import nbformat as nbf

from two_stage_pipeliner.core.reporter import Reporter
from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from two_stage_pipeliner.inference_models.detection.core import DetectionModelSpec
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.metrics.detection import get_df_detection_metrics, get_df_detection_recall_per_class
from two_stage_pipeliner.visualizers.detection import DetectionVisualizer
from two_stage_pipeliner.logging import logger

DETECTION_MODEL_SPEC_FILENAME = "detection_model_spec.pkl"
IMAGES_DATA_FILENAME = "images_data.pkl"


def detection_interactive_work(directory: Union[str, Path],
                               score_threshold: float,
                               minimum_iou: float):
    directory = Path(directory)
    model_spec_filepath = directory / DETECTION_MODEL_SPEC_FILENAME
    with open(model_spec_filepath, "rb") as src:
        model_spec = pickle.load(src)
    detection_model = model_spec.load()
    detection_inferencer = DetectionInferencer(detection_model)

    images_data_filepath = directory / IMAGES_DATA_FILENAME
    with open(images_data_filepath, "rb") as src:
        images_data = pickle.load(src)
    detection_visualizer = DetectionVisualizer(detection_inferencer)
    detection_visualizer.visualize(images_data, score_threshold=score_threshold,
                                   show_TP_FP_FN_with_minimum_iou=minimum_iou)


class DetectionReporter(Reporter):
    def _get_markdowns(self,
                       df_detection_metrics: pd.DataFrame,
                       df_detection_recall_per_class: pd.DataFrame) -> List[str]:
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
            f'{df_detection_metrics.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## General recall by class\n'
            f'{df_detection_recall_per_class.to_markdown(stralign="center")}''\n'
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

    def report(
        self,
        model_spec: DetectionModelSpec,
        output_directory: Union[str, Path],
        true_images_data: List[ImageData],
        score_threshold: float,
        minimum_iou: float,
        batch_size: int = 16
    ):

        model = model_spec.load()
        inferencer = DetectionInferencer(model)
        images_data_gen = BatchGeneratorImageData(true_images_data, batch_size=batch_size,
                                                  use_not_caught_elements_as_last_batch=True)
        pred_images_data = inferencer.predict(images_data_gen, score_threshold=score_threshold)
        raw_pred_images_data = inferencer.predict(images_data_gen, score_threshold=0.)
        df_detection_metrics = get_df_detection_metrics(true_images_data, pred_images_data, minimum_iou,
                                                        raw_pred_images_data)
        df_detection_recall_per_class = get_df_detection_recall_per_class(true_images_data, pred_images_data,
                                                                          minimum_iou)
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True, parents=True)
        model_spec_filepath = output_directory / DETECTION_MODEL_SPEC_FILENAME
        with open(model_spec_filepath, 'wb') as out:
            pickle.dump(model_spec, out)
        images_data_filepath = output_directory / IMAGES_DATA_FILENAME
        with open(images_data_filepath, 'wb') as out:
            pickle.dump(images_data_gen.data, out)

        markdowns = self._get_markdowns(df_detection_metrics, df_detection_recall_per_class)
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
        nbf.write(nb, str(output_directory / 'report.ipynb'))
        logger.info(f"Detection report saved to '{output_directory}'.")
