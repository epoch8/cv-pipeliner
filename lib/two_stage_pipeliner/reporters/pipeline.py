import pickle
from pathlib import Path
from typing import Union, List

import pandas as pd
import nbformat as nbf

from two_stage_pipeliner.core.reporter import Reporter
from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from two_stage_pipeliner.inference_models.pipeline import PipelineModelSpec
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer
from two_stage_pipeliner.metrics.pipeline import get_df_pipeline_metrics
from two_stage_pipeliner.visualizers.pipeline import PipelineVisualizer
from two_stage_pipeliner.logging import logger

PIPELINE_MODEL_SPEC_FILENAME = "pipeline_model_spec.pkl"
IMAGES_DATA_FILENAME = "images_data.pkl"


def pipeline_interactive_work(directory: Union[str, Path],
                              detection_score_threshold: float,
                              minimum_iou: float):
    directory = Path(directory)
    pipeline_model_spec_filepath = directory / PIPELINE_MODEL_SPEC_FILENAME
    with open(pipeline_model_spec_filepath, "rb") as src:
        pipeline_model_spec = pickle.load(src)
    pipeline_model = pipeline_model_spec.load()

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
        codes.append('''
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
''')
        codes.append(f'''
from two_stage_pipeliner.reporters.pipeline import pipeline_interactive_work
pipeline_interactive_work(
    directory='.',
    detection_score_threshold={detection_score_threshold},
    minimum_iou={minimum_iou}
)''')
        codes = [code.strip() for code in codes]
        return codes

    def report(
        self,
        model_spec: PipelineModelSpec,
        output_directory: Union[str, Path],
        true_images_data: List[ImageData],
        detection_score_threshold: float,
        minimum_iou: float,
        extra_bbox_label: str = None,
        batch_size: int = 16
    ):
        model = model_spec.load()
        inferencer = PipelineInferencer(model)
        images_data_gen = BatchGeneratorImageData(true_images_data, batch_size=batch_size,
                                                  use_not_caught_elements_as_last_batch=True)
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
            use_soft_metrics_with_known_labels=model.class_names
        )
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True, parents=True)
        pipeline_model_spec_filepath = output_directory / PIPELINE_MODEL_SPEC_FILENAME
        with open(pipeline_model_spec_filepath, 'wb') as out:
            pickle.dump(model_spec, out)
        images_data_filepath = output_directory / IMAGES_DATA_FILENAME
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
        nbf.write(nb, str(output_directory / 'report.ipynb'))
        logger.info(f"Pipeline report saved to '{output_directory}'.")
