import pickle
from pathlib import Path
from typing import Union, List

import pandas as pd
import nbformat as nbf

from two_stage_pipeliner.core.reporter import Reporter
from two_stage_pipeliner.core.data import BboxData
from two_stage_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from two_stage_pipeliner.inference_models.classification.core import ClassificationModelSpec
from two_stage_pipeliner.inferencers.classification import ClassificationInferencer
from two_stage_pipeliner.metrics.classification import get_df_classification_metrics
from two_stage_pipeliner.visualizers.classification import ClassificationVisualizer
from two_stage_pipeliner.logging import logger

CLASSIFICATION_MODEL_SPEC_FILENAME = "classification_model_spec.pkl"
BBOXES_DATA_FILENAME = "bboxes_data.pkl"


def classification_interactive_work(
    directory: Union[str, Path],
    use_all_data: bool = False
):
    directory = Path(directory)
    model_spec_filepath = directory / CLASSIFICATION_MODEL_SPEC_FILENAME
    with open(model_spec_filepath, "rb") as src:
        model_spec = pickle.load(src)
    classification_model = model_spec.load()
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
        codes.append('''
from two_stage_pipeliner.reporters.classification import classification_interactive_work
classification_interactive_work(directory='.', use_all_data=True)''')
        codes = [code.strip() for code in codes]
        return codes

    def report(
        self,
        model_spec: ClassificationModelSpec,
        output_directory: Union[str, Path],
        n_true_bboxes_data: List[List[BboxData]],
        batch_size: int = 16
    ):
        if hasattr(model_spec, 'preprocess_input'):
            assert (
                isinstance(model_spec.preprocess_input, str)
                or
                isinstance(model_spec.preprocess_input, Path)
            )
        model = model_spec.load()
        inferencer = ClassificationInferencer(model)
        bboxes_data_gen = BatchGeneratorBboxData(n_true_bboxes_data,
                                                 batch_size=batch_size,
                                                 use_not_caught_elements_as_last_batch=True)
        n_pred_bboxes_data = inferencer.predict(bboxes_data_gen)
        df_classification_metrics = get_df_classification_metrics(n_true_bboxes_data, n_pred_bboxes_data)
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True, parents=True)
        model_spec_filepath = output_directory / CLASSIFICATION_MODEL_SPEC_FILENAME
        with open(model_spec_filepath, 'wb') as out:
            pickle.dump(model_spec, out)
        bboxes_data_filepath = output_directory / BBOXES_DATA_FILENAME
        with open(bboxes_data_filepath, 'wb') as out:
            pickle.dump(n_true_bboxes_data, out)

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
        nbf.write(nb, str(output_directory / 'report.ipynb'))
        logger.info(f"Classification report saved to '{output_directory}'.")
