import pickle
from pathlib import Path
from typing import Union, List

import pandas as pd
import nbformat as nbf

from two_stage_pipeliner.core.reporter import Reporter
from two_stage_pipeliner.core.batch_generator import BatchGeneratorBboxData
from two_stage_pipeliner.inferencers.classification import ClassificationInferencer
from two_stage_pipeliner.metrics_counters.classification import ClassificationMetricsCounter
from two_stage_pipeliner.visualizers.classification import ClassificationVisualizer
from two_stage_pipeliner.inference_models.classification.tf.classifier import ClassifierTF

from two_stage_pipeliner.logging import logger

CHECKPOINT_FILENAME = "checkpoint.pkl"
BBOXES_DATA_FILENAME = "bboxes_data.pkl"


def classification_interactive_work(directory: Union[str, Path],
                                    use_all_data: bool = False):
    directory = Path(directory)
    checkpoint_filepath = directory / CHECKPOINT_FILENAME
    with open(checkpoint_filepath, "rb") as src:
        checkpoint = pickle.load(src)
    classification_model = ClassifierTF()
    classification_model.load(checkpoint)

    classification_inferencer = ClassificationInferencer(classification_model)

    images_data_filepath = directory / BBOXES_DATA_FILENAME
    with open(images_data_filepath, "rb") as src:
        n_bboxes_data = pickle.load(src)

    classification_visualizer = ClassificationVisualizer(classification_inferencer)
    classification_visualizer.visualize(n_bboxes_data, use_all_data=use_all_data, show_TP_FP=True)


class ClassificationReporter(Reporter):
    def _get_markdowns(self,
                       df_classification_metrics: pd.DataFrame) -> List[str]:
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
            '## Common classification metrics\n'
            f'{df_classification_metrics.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Interactive work:\n'
        )
        return markdowns

    def _get_codes(self) -> List[str]:
        codes = []
        codes.append(f'''
from two_stage_pipeliner.reporters.classification import classification_interactive_work
classification_interactive_work(directory='.', use_all_data=True)''')
        codes = [code.strip() for code in codes]
        return codes

    def report(self,
               inferencer: ClassificationInferencer,
               data_generator: BatchGeneratorBboxData,
               directory: Union[str, Path]):

        metrics_counter = ClassificationMetricsCounter(inferencer)
        df_classification_metrics = metrics_counter.score(data_generator)
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        checkpoint_filepath = directory / CHECKPOINT_FILENAME
        with open(checkpoint_filepath, 'wb') as out:
            pickle.dump(inferencer.model.checkpoint, out)
        bboxes_data_filepath = directory / BBOXES_DATA_FILENAME
        with open(bboxes_data_filepath, 'wb') as out:
            pickle.dump(data_generator.data, out)

        markdowns = self._get_markdowns(df_classification_metrics)
        codes = self._get_codes()

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
        logger.info(f"Classification report saved to '{directory}'.")
