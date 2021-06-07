import pickle
from pathlib import Path
from typing import Union, List, Tuple
from dataclasses import dataclass

import pandas as pd
import nbformat as nbf

from cv_pipeliner.core.reporter import Reporter
from cv_pipeliner.core.data import ImageData
from cv_pipeliner.batch_generators.image_data import BatchGeneratorImageData
from cv_pipeliner.inference_models.detection.core import DetectionModelSpec
from cv_pipeliner.inferencers.detection import DetectionInferencer
from cv_pipeliner.metrics.detection import (
    get_df_detection_metrics, get_df_detection_recall_per_class,
    df_detection_recall_per_class_columns
)
from cv_pipeliner.visualizers.detection import DetectionVisualizer
from cv_pipeliner.logging import logger
from cv_pipeliner.utils.dataframes import transpose_columns_and_write_diffs_to_df_with_tags

DETECTION_MODEL_SPEC_PREFIX = "detection_model_spec"
IMAGES_DATA_FILENAME = "images_data.pkl"


def detection_interactive_work(
    directory: Union[str, Path],
    tag: str,
    score_threshold: float,
    minimum_iou: float
):
    directory = Path(directory)
    detection_model_spec_filepath = directory / f"{DETECTION_MODEL_SPEC_PREFIX}_{tag}.pkl"
    with open(detection_model_spec_filepath, "rb") as src:
        detection_model_spec = pickle.load(src)
    detection_model = detection_model_spec.load()
    detection_inferencer = DetectionInferencer(detection_model)

    images_data_filepath = directory / IMAGES_DATA_FILENAME
    with open(images_data_filepath, "rb") as src:
        images_data = pickle.load(src)
    detection_visualizer = DetectionVisualizer(detection_inferencer)
    detection_visualizer.visualize(
        images_data=images_data,
        score_threshold=score_threshold,
        show_TP_FP_FN_with_minimum_iou=minimum_iou
    )


@dataclass
class DetectionReportData:
    df_detection_metrics: pd.DataFrame = None
    df_detection_recall_per_class: pd.DataFrame = None
    df_detection_metrics_short: pd.DataFrame = None
    tag: str = None

    def __init__(
        self,
        df_detection_metrics: pd.DataFrame = None,
        df_detection_recall_per_class: pd.DataFrame = None,
        tag: str = None,
        collect_the_rest: bool = True
    ):
        self.df_detection_metrics = df_detection_metrics.copy()
        self.df_detection_recall_per_class = df_detection_recall_per_class.copy()

        if not collect_the_rest:
            return

        self.df_detection_metrics_short = df_detection_metrics.loc[['precision', 'recall']].copy()
        self.tag = tag
        if tag is not None:
            for df in self.get_all_dfs():
                df.columns = [f"{column} [{tag}]" for column in df.columns]

    def get_all_dfs(self) -> List[pd.DataFrame]:
        return [
            self.df_detection_metrics,
            self.df_detection_recall_per_class,
            self.df_detection_metrics_short,
        ]


def concat_detections_reports_datas(
    detections_reports_datas: List[DetectionReportData],
    compare_tag: str = None
) -> DetectionReportData:
    tags = [classification_report_data.tag for classification_report_data in detections_reports_datas]
    df_detection_metrics = transpose_columns_and_write_diffs_to_df_with_tags(
        pd.concat(
            [
                tag_detection_report_data.df_detection_metrics
                for tag_detection_report_data in detections_reports_datas
            ],
            axis=1
        ),
        columns=['value'],
        tags=tags,
        compare_tag=compare_tag
    )
    df_detection_recall_per_class = transpose_columns_and_write_diffs_to_df_with_tags(
        pd.concat(
            [
                tag_detection_report_data.df_detection_recall_per_class
                for tag_detection_report_data in detections_reports_datas
            ],
            axis=1
        ),
        columns=df_detection_recall_per_class_columns,
        tags=tags,
        compare_tag=compare_tag
    )
    detection_report_data = DetectionReportData(
        df_detection_metrics=df_detection_metrics,
        df_detection_recall_per_class=df_detection_recall_per_class,
        collect_the_rest=False
    )
    detection_report_data.df_detection_metrics_short = transpose_columns_and_write_diffs_to_df_with_tags(
        df_with_tags=pd.concat(
            [
                tag_detection_report_data.df_detection_metrics_short
                for tag_detection_report_data in detections_reports_datas
            ],
            axis=1
        ),
        columns=['value'],
        tags=tags,
        compare_tag=compare_tag
    )
    return detection_report_data


class DetectionReporter(Reporter):
    def _get_markdowns(
        self,
        detection_report_data: DetectionReportData
    ) -> List[str]:
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
            '## General detector metrics\n'
            f'{detection_report_data.df_detection_metrics_short.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '---'
        )
        markdowns.append(
            '## Common detector metrics\n'
            f'{detection_report_data.df_detection_metrics.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '## Recall by class\n'
            f'{detection_report_data.df_detection_recall_per_class.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '---'
        )
        markdowns.append(
            '## Interactive work:\n'
        )
        return markdowns

    def _get_codes(
        self,
        tags: List[str],
        scores_thresholds: List[float],
        minimum_iou: float
    ) -> List[str]:
        codes = []
        codes.append('''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
''')
        codes.append('''
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
''')
        for tag, score_threshold in zip(tags, scores_thresholds):
            codes.append(f'''
from cv_pipeliner.reporters.detection import detection_interactive_work
detection_interactive_work(
    directory='.',
    tag='{tag}',
    score_threshold={score_threshold},
    minimum_iou={minimum_iou}
)''')
        codes = [code.strip() for code in codes]
        return codes

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
        raw_pred_images_data = inferencer.predict(images_data_gen, score_threshold=0.)
        pred_images_data = [
            ImageData(
                image_path=image_data.image_path,
                bboxes_data=[
                    bbox_data
                    for bbox_data in image_data.bboxes_data
                    if bbox_data.detection_score >= score_threshold
                ]
            )
            for image_data in raw_pred_images_data
        ]
        df_detection_metrics = get_df_detection_metrics(
            true_images_data=true_images_data,
            pred_images_data=pred_images_data,
            minimum_iou=minimum_iou,
            raw_pred_images_data=raw_pred_images_data
        )
        df_detection_recall_per_class = get_df_detection_recall_per_class(
            true_images_data=true_images_data,
            pred_images_data=pred_images_data,
            minimum_iou=minimum_iou,
        )
        return df_detection_metrics, df_detection_recall_per_class

    def _save_report(
        self,
        models_specs: List[DetectionModelSpec],
        tags: List[str],
        output_directory: Union[str, Path],
        true_images_data: List[ImageData],
        markdowns: List[str],
        codes: List[str],
    ):
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True, parents=True)
        for model_spec, tag in zip(models_specs, tags):
            if model_spec is None:
                continue
            detection_model_spec_filepath = output_directory / f"{DETECTION_MODEL_SPEC_PREFIX}_{tag}.pkl"
            with open(detection_model_spec_filepath, 'wb') as out:
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
        logger.info(f"Detection report saved to '{output_directory}'.")

    def report(
        self,
        models_specs: List[DetectionModelSpec],
        tags: List[str],
        scores_thresholds: List[float],
        compare_tag: str,
        output_directory: Union[str, Path],
        true_images_data: List[ImageData],
        minimum_iou: float,
        batch_size: int = 16
    ):
        assert len(models_specs) == len(tags)
        assert len(tags) == len(scores_thresholds)
        assert compare_tag in tags

        detections_reports_datas = []
        for model_spec, tag, score_threshold in zip(models_specs, tags, scores_thresholds):
            logger.info(f"Making inference and counting metrics for '{tag}'...")
            tag_df_detection_metrics, tag_df_detection_recall_per_class = self._inference_detection_and_get_metrics(
                model_spec=model_spec,
                true_images_data=true_images_data,
                score_threshold=score_threshold,
                minimum_iou=minimum_iou,
                batch_size=batch_size
            )

            detections_reports_datas.append(DetectionReportData(
                df_detection_metrics=tag_df_detection_metrics,
                df_detection_recall_per_class=tag_df_detection_recall_per_class,
                tag=tag
            ))

        detection_report_data = concat_detections_reports_datas(
            detections_reports_datas=detections_reports_datas,
            compare_tag=compare_tag
        )
        markdowns = self._get_markdowns(
            detection_report_data=detection_report_data,
        )
        codes = self._get_codes(
            tags=tags,
            scores_thresholds=scores_thresholds,
            minimum_iou=minimum_iou,
        )
        self._save_report(
            models_specs=models_specs,
            tags=tags,
            output_directory=output_directory,
            true_images_data=true_images_data,
            markdowns=markdowns,
            codes=codes
        )

    def report_on_predictions(
        self,
        true_images_data: List[ImageData],
        pred_images_data: List[ImageData],
        raw_pred_images_data: List[ImageData],
        tag: str,
        output_directory: Union[str, Path],
        minimum_iou: float,
    ):

        logger.info(f"Counting metrics for '{tag}'...")
        tag_df_detection_metrics = get_df_detection_metrics(
            true_images_data=true_images_data,
            pred_images_data=pred_images_data,
            minimum_iou=minimum_iou,
            raw_pred_images_data=raw_pred_images_data
        )
        tag_df_detection_recall_per_class = get_df_detection_recall_per_class(
            true_images_data=true_images_data,
            pred_images_data=pred_images_data,
            minimum_iou=minimum_iou,
        )
        detections_reports_datas = [DetectionReportData(
            df_detection_metrics=tag_df_detection_metrics,
            df_detection_recall_per_class=tag_df_detection_recall_per_class,
            tag=tag
        )]

        detection_report_data = concat_detections_reports_datas(
            detections_reports_datas=detections_reports_datas,
            compare_tag=tag
        )
        markdowns = self._get_markdowns(
            detection_report_data=detection_report_data,
        )
        self._save_report(
            models_specs=[None],
            tags=[tag],
            output_directory=output_directory,
            true_images_data=true_images_data,
            markdowns=markdowns,
            codes=[]
        )
