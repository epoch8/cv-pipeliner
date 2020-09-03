import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Tuple

import pandas as pd
import nbformat as nbf

from two_stage_pipeliner.core.reporter import Reporter
from two_stage_pipeliner.core.data import ImageData
from two_stage_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from two_stage_pipeliner.inference_models.detection.core import DetectionModelSpec
from two_stage_pipeliner.inference_models.pipeline import PipelineModelSpec
from two_stage_pipeliner.inferencers.detection import DetectionInferencer
from two_stage_pipeliner.inferencers.pipeline import PipelineInferencer
from two_stage_pipeliner.metrics.detection import get_df_detection_metrics
from two_stage_pipeliner.metrics.pipeline import get_df_pipeline_metrics
from two_stage_pipeliner.visualizers.pipeline import PipelineVisualizer
from two_stage_pipeliner.logging import logger

PIPELINE_MODEL_SPEC_NAME = "pipeline_model_spec"
IMAGES_DATA_FILENAME = "images_data.pkl"


def pipeline_interactive_work(
    directory: Union[str, Path],
    detection_score_threshold: float,
    minimum_iou: float,
    tag: str
):
    directory = Path(directory)
    pipeline_model_spec_filepath = directory / f"{PIPELINE_MODEL_SPEC_NAME}_{tag}.pkl"
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


@dataclass
class PipelineReportData:
    df_detection_metrics: Union[pd.DataFrame, List[pd.DataFrame]] = None
    df_pipeline_metrics_strict: Union[pd.DataFrame, List[pd.DataFrame]] = None
    df_pipeline_metrics_soft: Union[pd.DataFrame, List[pd.DataFrame]] = None
    df_detection_metrics_short: pd.DataFrame = None
    df_pipeline_metrics_short: pd.DataFrame = None
    df_correct_preds: pd.DataFrame = None
    df_incorrect_preds: pd.DataFrame = None
    df_incorrect_preds_percentage: pd.DataFrame = None
    tag: str = None

    def __init__(
        self,
        df_detection_metrics: pd.DataFrame,
        df_pipeline_metrics_strict: pd.DataFrame,
        df_pipeline_metrics_soft: pd.DataFrame,
        tag: str = None,
    ):
        if all(df is None for df in [df_detection_metrics, df_pipeline_metrics_strict, df_pipeline_metrics_soft]):
            return
        self.df_detection_metrics = df_detection_metrics.copy()
        self.df_pipeline_metrics_strict = df_pipeline_metrics_strict.copy()
        self.df_pipeline_metrics_soft = df_pipeline_metrics_soft.copy()
        self.df_detection_metrics_short = df_detection_metrics.loc[['precision', 'recall']].copy()
        self.df_pipeline_metrics_short = self._get_df_pipeline_metrics_short(
            df_pipeline_metrics_strict=df_pipeline_metrics_strict,
            df_pipeline_metrics_soft=df_pipeline_metrics_soft
        )
        self.df_correct_preds = pd.DataFrame({
            'TP (detector)': [df_detection_metrics.loc['TP', 'value']],
            'TP (pipeline)': [df_pipeline_metrics_strict.loc['micro_average', 'TP']]
        }, index=['value'], dtype=int).T
        self.df_incorrect_preds = self._get_df_incorrect_preds(
            df_detection_metrics=df_detection_metrics,
            df_pipeline_metrics_strict=df_pipeline_metrics_strict,
            df_pipeline_metrics_soft=df_pipeline_metrics_soft
        )
        self.df_incorrect_preds_percentage = (
            self.df_incorrect_preds[['value']] / self.df_incorrect_preds['value'].sum() * 100
        ).round(1)
        self.df_incorrect_preds_percentage['value'] = [f"{value}%" for value in self.df_incorrect_preds_percentage['value']]

        self.tag = tag
        if tag is not None:
            for df in [
                self.df_detection_metrics, self.df_pipeline_metrics_strict, self.df_pipeline_metrics_soft,
                self.df_detection_metrics_short,
                self.df_pipeline_metrics_short, self.df_correct_preds, self.df_incorrect_preds,
                self.df_incorrect_preds_percentage
            ]:
                self._give_tag_to_df(tag, df)

    def _give_tag_to_df(self, tag: str, df: pd.DataFrame):
        df.columns = [f"{column} [{tag}]" for column in df.columns]

    def _get_df_pipeline_metrics_short(
        self,
        df_pipeline_metrics_strict: pd.DataFrame,
        df_pipeline_metrics_soft: pd.DataFrame
    ) -> pd.DataFrame:
        df_strict = df_pipeline_metrics_strict.loc[['weighted_average'], ['precision', 'recall']]
        df_soft = df_pipeline_metrics_soft.loc[['weighted_average'], ['precision', 'recall']]
        df_pipeline_metrics_short = pd.concat([df_strict, df_soft], axis=1).T
        df_pipeline_metrics_short.index = [
            'precision (strict)', 'recall (strict)', 'precision (soft)', 'recall (soft)'
        ]
        df_pipeline_metrics_short.columns = ['value']
        return df_pipeline_metrics_short

    def _get_df_incorrect_preds(
        self,
        df_detection_metrics: pd.DataFrame,
        df_pipeline_metrics_strict: pd.DataFrame,
        df_pipeline_metrics_soft: pd.DataFrame
    ) -> pd.DataFrame:
        df_all_class_names = df_pipeline_metrics_soft[df_pipeline_metrics_soft['is known by classifier'].notna()]
        uknown_classes = df_all_class_names[~df_all_class_names['is known by classifier'].astype(bool)].index
        df_incorrect_preds = pd.DataFrame({
            'FP (detector)': [df_detection_metrics.loc['FP', 'value']],
            'FN (detector)': [df_detection_metrics.loc['FN', 'value']],
            'FP (classifier, known classes)': [df_pipeline_metrics_soft.loc['micro_average', 'FP']],
            'FP (classifier, unknown classes)': [df_pipeline_metrics_strict.loc[uknown_classes, 'FP'].sum()]
        }, index=['value'], dtype=int).T
        return df_incorrect_preds


def concat_pipelines_reports_datas(
    pipelines_reports_datas: List[PipelineReportData],
    tags: List[str],
    compare_tag: str = None
):
    pipeline_report_data = PipelineReportData(  # TODO: make concat for these metrics
        df_detection_metrics=None,
        df_pipeline_metrics_strict=None,
        df_pipeline_metrics_soft=None
    )
    pipeline_report_data.df_detection_metrics_short = pd.concat(
        [pipeline_report_data.df_detection_metrics_short for pipeline_report_data in pipelines_reports_datas],
        axis=1
    )
    pipeline_report_data.df_pipeline_metrics_short = pd.concat(
        [pipeline_report_data.df_pipeline_metrics_short for pipeline_report_data in pipelines_reports_datas],
        axis=1
    )
    pipeline_report_data.df_correct_preds = pd.concat(
        [pipeline_report_data.df_correct_preds for pipeline_report_data in pipelines_reports_datas],
        axis=1
    )
    pipeline_report_data.df_incorrect_preds = pd.concat(
        [pipeline_report_data.df_incorrect_preds for pipeline_report_data in pipelines_reports_datas],
        axis=1
    )
    pipeline_report_data.df_incorrect_preds_percentage = pd.concat(
        [pipeline_report_data.df_incorrect_preds_percentage for pipeline_report_data in pipelines_reports_datas],
        axis=1
    )
    if compare_tag is not None:
        for df in [
            pipeline_report_data.df_detection_metrics_short, pipeline_report_data.df_pipeline_metrics_short,
            pipeline_report_data.df_correct_preds, pipeline_report_data.df_incorrect_preds
        ]:
            df[f'value [{compare_tag}]'] = [round(value, 3) for value in df[f'value [{compare_tag}]']]
            compare_tag_values = df[f'value [{compare_tag}]']
            for tag in tags:
                if tag == compare_tag:
                    continue
                tag_values = df[f'value [{tag}]']
                diffs = ((tag_values - compare_tag_values) / compare_tag_values)
                suffixes = ['+' if diff > 0 else '' for diff in diffs]
                df[f'value [{tag}]'] = [
                    f"{round(value, 3)} ({suffix}{int(round(100 * diff))}%)"
                    for value, suffix, diff in zip(tag_values, suffixes, diffs)
                ]

    # TODO: make concat for these metrics
    pipeline_report_data.df_detection_metrics = [
        pipeline_report_data.df_detection_metrics for pipeline_report_data in pipelines_reports_datas
    ]
    pipeline_report_data.df_pipeline_metrics_strict = [
        pipeline_report_data.df_pipeline_metrics_strict for pipeline_report_data in pipelines_reports_datas
    ]
    pipeline_report_data.df_pipeline_metrics_soft = [
        pipeline_report_data.df_pipeline_metrics_soft for pipeline_report_data in pipelines_reports_datas
    ]

    return pipeline_report_data


class PipelineReporter(Reporter):
    def _get_markdowns(
        self,
        pipeline_report_data: PipelineReportData,
        tags: List[str] = None
    ) -> List[str]:

        if tags is not None:
            assert len(pipeline_report_data.df_pipeline_metrics_strict) == len(tags)
            assert len(pipeline_report_data.df_pipeline_metrics_soft) == len(tags)

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
            '## Detection metrics\n'
            f'{pipeline_report_data.df_detection_metrics_short.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Pipeline metrics (weighted average)\n'
            f'{pipeline_report_data.df_pipeline_metrics_short.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Correct pipeline predictions in absolute values\n'
            f'{pipeline_report_data.df_correct_preds.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Main pipeline errors in absolute values\n'
            f'{pipeline_report_data.df_incorrect_preds.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Percentage of errors\n'
            f'{pipeline_report_data.df_incorrect_preds_percentage.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '---'
        )
        if tags is not None:
            for tag, tag_df_detection_metrics, tag_df_pipeline_metrics_strict, tag_df_pipeline_metrics_soft in zip(
                tags,
                pipeline_report_data.df_detection_metrics,
                pipeline_report_data.df_pipeline_metrics_strict,
                pipeline_report_data.df_pipeline_metrics_soft
            ):
                markdowns.append(
                    f'## Pipeline [{tag}]''\n'
                )
                markdowns.append(
                    f'## Detection metrics of [{tag}]:''\n'
                    f'{tag_df_detection_metrics.to_markdown(stralign="center")}''\n'
                )
                markdowns.append(
                    f'## Pipeline metrics of [{tag}] (strict):''\n'
                    f'{tag_df_pipeline_metrics_strict.to_markdown(stralign="center")}''\n'
                )
                markdowns.append(
                    f'## Pipeline metrics of [{tag}] (soft):''\n'
                    f'{tag_df_pipeline_metrics_soft.to_markdown(stralign="center")}''\n'
                )
                markdowns.append(
                    '---'
                )
        else:
            markdowns.append(
                f'## Detection metrics:''\n'
                f'{pipeline_report_data.df_detection_metrics.to_markdown(stralign="center")}''\n'
            )
            markdowns.append(
                f'## Pipeline metrics (strict):''\n'
                f'{pipeline_report_data.df_pipeline_metrics_strict.to_markdown(stralign="center")}''\n'
            )
            markdowns.append(
                f'## Pipeline metrics (soft):''\n'
                f'{pipeline_report_data.df_pipeline_metrics_soft.to_markdown(stralign="center")}''\n'
            )
        markdowns.append(
            '## Interactive work:\n'
        )
        return markdowns

    def _get_codes(
        self,
        tags: List[str],
        detection_scores_thresholds: List[float],
        minimum_iou: float,
    ) -> List[str]:

        assert len(detection_scores_thresholds) == len(tags)

        codes = []
        codes.append('''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
''')
        codes.append('''
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
''')
        codes.append('''
from two_stage_pipeliner.reporters.pipeline import pipeline_interactive_work
''')
        for detection_score_threshold, tag in zip(detection_scores_thresholds, tags):
            codes.append(f'''
pipeline_interactive_work(
    directory='.',
    detection_score_threshold={detection_score_threshold},
    minimum_iou={minimum_iou},
    tag='{tag}'
)''')
        codes = [code.strip() for code in codes]
        return codes

    def _inference_pipeline_and_get_metrics(
        self,
        model_spec: PipelineModelSpec,
        true_images_data: List[ImageData],
        detection_score_threshold: float,
        minimum_iou: float,
        extra_bbox_label: str = None,
        batch_size: int = 16
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        return df_pipeline_metrics, df_pipeline_metrics_soft

    def _inference_detection_and_get_metrics(
        self,
        model_spec: DetectionModelSpec,
        true_images_data: List[ImageData],
        score_threshold: float,
        minimum_iou: float,
        extra_bbox_label: str = None,
        batch_size: int = 16
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        detection_model = model_spec.load()
        inferencer = DetectionInferencer(detection_model)
        images_data_gen = BatchGeneratorImageData(true_images_data, batch_size=batch_size,
                                                  use_not_caught_elements_as_last_batch=True)
        pred_images_data = inferencer.predict(images_data_gen, score_threshold=score_threshold)
        raw_pred_images_data = inferencer.predict(images_data_gen, score_threshold=0.)
        df_detection_metrics = get_df_detection_metrics(
            true_images_data=true_images_data,
            pred_images_data=pred_images_data,
            minimum_iou=minimum_iou,
            raw_pred_images_data=raw_pred_images_data
        )
        return df_detection_metrics

    def _save_report(
        self,
        models_specs: List[PipelineModelSpec],
        tags: List[str],
        output_directory: Union[str, Path],
        true_images_data: List[ImageData],
        markdowns: List[str],
        codes: List[str],
    ):
        assert len(models_specs) == len(tags)

        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True, parents=True)
        for model_spec, tag in zip(models_specs, tags):
            pipeline_model_spec_filepath = output_directory / f"{PIPELINE_MODEL_SPEC_NAME}_{tag}.pkl"
            with open(pipeline_model_spec_filepath, 'wb') as out:
                pickle.dump(model_spec, out)
        images_data_filepath = output_directory / IMAGES_DATA_FILENAME
        with open(images_data_filepath, 'wb') as out:
            pickle.dump(true_images_data, out)

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
        if hasattr(model_spec.classification_model_spec, 'preprocess_input'):
            assert (
                isinstance(model_spec.classification_model_spec.preprocess_input, str)
                or
                isinstance(model_spec.classification_model_spec.preprocess_input, Path)
            )
        df_detection_metrics = self._inference_detection_and_get_metrics(
            model_spec=model_spec.detection_model_spec,
            true_images_data=true_images_data,
            score_threshold=detection_score_threshold,
            minimum_iou=minimum_iou,
            batch_size=batch_size
        )
        df_pipeline_metrics_strict, df_pipeline_metrics_soft = self._inference_pipeline_and_get_metrics(
            model_spec=model_spec,
            true_images_data=true_images_data,
            detection_score_threshold=detection_score_threshold,
            minimum_iou=minimum_iou,
            extra_bbox_label=extra_bbox_label,
            batch_size=batch_size
        )

        pipeline_report_data = PipelineReportData(
            df_detection_metrics=df_detection_metrics,
            df_pipeline_metrics_strict=df_pipeline_metrics_strict,
            df_pipeline_metrics_soft=df_pipeline_metrics_soft
        )

        tags = ['main']

        markdowns = self._get_markdowns(pipeline_report_data=pipeline_report_data)
        codes = self._get_codes(
            tags=tags,
            detection_scores_thresholds=[detection_score_threshold],
            minimum_iou=minimum_iou
        )
        self._save_report(
            models_specs=[model_spec],
            tags=tags,
            output_directory=output_directory,
            true_images_data=true_images_data,
            markdowns=markdowns,
            codes=codes
        )

    def report_many(
        self,
        models_specs: List[PipelineModelSpec],
        tags: List[str],
        detection_scores_thresholds: float,
        extra_bbox_labels: List[str],
        compare_tag: str,
        output_directory: Union[str, Path],
        true_images_data: List[ImageData],
        minimum_iou: float,
        batch_size: int = 16
    ):
        assert len(models_specs) == len(tags)
        assert len(tags) == len(detection_scores_thresholds)
        assert len(detection_scores_thresholds) == len(extra_bbox_labels)
        assert compare_tag in tags

        pipelines_reports_datas = []
        for model_spec, tag, detection_score_threshold, extra_bbox_label in zip(
            models_specs, tags, detection_scores_thresholds, extra_bbox_labels
        ):
            logger.info(f"Making inference and counting metrics for '{tag}'...")
            tag_df_detection_metrics = self._inference_detection_and_get_metrics(
                model_spec=model_spec.detection_model_spec,
                true_images_data=true_images_data,
                score_threshold=detection_score_threshold,
                minimum_iou=minimum_iou,
                batch_size=batch_size
            )
            tag_df_pipeline_metrics_strict, tag_df_pipeline_metrics_soft = self._inference_pipeline_and_get_metrics(
                model_spec=model_spec,
                true_images_data=true_images_data,
                detection_score_threshold=detection_score_threshold,
                minimum_iou=minimum_iou,
                extra_bbox_label=extra_bbox_label,
                batch_size=batch_size
            )

            pipelines_reports_datas.append(PipelineReportData(
                df_detection_metrics=tag_df_detection_metrics,
                df_pipeline_metrics_strict=tag_df_pipeline_metrics_strict,
                df_pipeline_metrics_soft=tag_df_pipeline_metrics_soft,
                tag=tag
            ))

        pipeline_report_data = concat_pipelines_reports_datas(
            pipelines_reports_datas=pipelines_reports_datas,
            tags=tags,
            compare_tag=compare_tag
        )
        markdowns = self._get_markdowns(
            pipeline_report_data=pipeline_report_data,
            tags=tags
        )
        codes = self._get_codes(
            tags=tags,
            detection_scores_thresholds=detection_scores_thresholds,
            minimum_iou=minimum_iou
        )
        self._save_report(
            models_specs=models_specs,
            tags=tags,
            output_directory=output_directory,
            true_images_data=true_images_data,
            markdowns=markdowns,
            codes=codes
        )
