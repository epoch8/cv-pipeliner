import pickle
from pathlib import Path
from typing import Union, List
from dataclasses import dataclass

import pandas as pd
import nbformat as nbf

from cv_pipeliner.core.reporter import Reporter
from cv_pipeliner.core.data import BboxData
from cv_pipeliner.batch_generators.bbox_data import BatchGeneratorBboxData
from cv_pipeliner.inference_models.classification.core import ClassificationModelSpec
from cv_pipeliner.inferencers.classification import ClassificationInferencer
from cv_pipeliner.metrics.classification import get_df_classification_metrics
from cv_pipeliner.visualizers.classification import ClassificationVisualizer
from cv_pipeliner.logging import logger
from cv_pipeliner.utils.dataframes import transpose_columns_and_write_diffs_to_df_with_tags

CLASSIFICATION_MODEL_SPEC_PREFIX = "classification_model_spec"
BBOXES_DATA_FILENAME = "n_bboxes_data.pkl"
CLASSIFICATIONS_REPORTS_DATAS_FILENAME = "classifications_reports_data.pkl"


def classification_interactive_work(
    directory: Union[str, Path],
    tag: str,
    use_all_data: bool = False,
):
    directory = Path(directory)
    classification_model_spec_filepath = directory / f"{CLASSIFICATION_MODEL_SPEC_PREFIX}_{tag}.pkl"
    with open(classification_model_spec_filepath, "rb") as src:
        classification_model_spec = pickle.load(src)
    classification_model = classification_model_spec.load()
    classification_inferencer = ClassificationInferencer(classification_model)

    bboxes_data_filepath = directory / BBOXES_DATA_FILENAME
    with open(bboxes_data_filepath, "rb") as src:
        n_bboxes_data = pickle.load(src)

    classification_visualizer = ClassificationVisualizer(classification_inferencer)
    classification_visualizer.visualize(
        n_bboxes_data=n_bboxes_data,
        use_all_data=use_all_data,
        show_TP_FP=True
    )


@dataclass
class ClassificationReportData:
    df_classification_metrics: pd.DataFrame = None
    df_classification_metrics_short: pd.DataFrame = None
    tag: str = None

    def __init__(
        self,
        df_classification_metrics: pd.DataFrame = None,
        tops_n: List[int] = [1],
        tag: str = None,
        collect_the_rest: bool = True,
    ):
        self.df_classification_metrics = df_classification_metrics.copy()
        if not collect_the_rest:
            return
        df_classification_metrics_MES_column = (
            ['mean_expected_steps'] if 'mean_expected_steps' in df_classification_metrics.columns else []
        )
        df_classification_metrics_short_columns = ['precision', 'recall'] + df_classification_metrics_MES_column + [
            item for sublist in [[f'precision@{top_n}', f'recall@{top_n}'] for top_n in tops_n if top_n > 1]
            for item in sublist
        ]
        self.df_classification_metrics_short = df_classification_metrics.loc[
            [
                'all_weighted_average', 'all_weighted_average_without_pseudo_classes',
                'known_weighted_average', 'known_weighted_average_without_pseudo_classes',
            ], df_classification_metrics_short_columns
        ].copy()

        self.tag = tag
        if tag is not None:
            for df in self.get_all_dfs():
                df.columns = [f"{column} [{tag}]" for column in df.columns]

    def get_all_dfs(self) -> List[pd.DataFrame]:
        return [
            self.df_classification_metrics,
            self.df_classification_metrics_short,
        ]


def concat_classifications_reports_datas(
    classifications_reports_datas: List[ClassificationReportData],
    df_classification_metrics_columns: List[str],
    tops_n: List[int],
    compare_tag: str = None,
) -> ClassificationReportData:
    tags = [classification_report_data.tag for classification_report_data in classifications_reports_datas]
    df_classification_metrics = transpose_columns_and_write_diffs_to_df_with_tags(
        pd.concat(
            [
                tag_classification_report_data.df_classification_metrics
                for tag_classification_report_data in classifications_reports_datas
            ],
            axis=1
        ),
        columns=df_classification_metrics_columns,
        tags=tags,
        compare_tag=compare_tag
    )
    classification_report_data = ClassificationReportData(
        df_classification_metrics=df_classification_metrics,
        tops_n=tops_n,
        collect_the_rest=False
    )
    classification_report_data.df_classification_metrics_short = transpose_columns_and_write_diffs_to_df_with_tags(
        df_with_tags=pd.concat(
            [
                tag_classification_report_data.df_classification_metrics_short
                for tag_classification_report_data in classifications_reports_datas
            ],
            axis=1
        ),
        columns=['precision', 'recall'],
        tags=tags,
        compare_tag=compare_tag
    )
    return classification_report_data


class ClassificationReporter(Reporter):
    def _get_markdowns(
        self,
        classification_report_data: ClassificationReportData,
        tags: List[str] = None
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
            '## General classification metrics\n'
            f'{classification_report_data.df_classification_metrics_short.to_markdown(stralign="center")}''\n'
        )
        markdowns.append(
            '---'
        )
        markdowns.append(
            '## Classification metrics\n'
            f'{classification_report_data.df_classification_metrics.to_markdown(stralign="center")}''\n'
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
        tags: List[str]
    ) -> List[str]:
        codes = []
        codes.append('''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
''')
        for tag in tags:
            codes.append(f'''
from cv_pipeliner.reporters.classification import classification_interactive_work
classification_interactive_work(
    directory='.',
    tag='{tag}',
    use_all_data=True
)''')

        codes = [code.strip() for code in codes]
        return codes

    def _inference_classification_and_get_metrics(
        self,
        model_spec: ClassificationModelSpec,
        n_true_bboxes_data: List[List[BboxData]],
        tops_n: List[int],
        batch_size: int,
        pseudo_class_names: List[str],
        step_penalty: int
    ) -> pd.DataFrame:
        classification_model = model_spec.load()
        inferencer = ClassificationInferencer(classification_model)
        bboxes_data_gen = BatchGeneratorBboxData(n_true_bboxes_data,
                                                 batch_size=batch_size,
                                                 use_not_caught_elements_as_last_batch=True)
        n_pred_bboxes_data = inferencer.predict(bboxes_data_gen, top_n=max(tops_n))
        df_classification_metrics = get_df_classification_metrics(
            n_true_bboxes_data=n_true_bboxes_data,
            n_pred_bboxes_data=n_pred_bboxes_data,
            pseudo_class_names=pseudo_class_names,
            known_class_names=classification_model.class_names,
            tops_n=tops_n,
            step_penalty=step_penalty
        )

        return df_classification_metrics

    def _save_report(
        self,
        models_specs: List[ClassificationModelSpec],
        tags: List[str],
        output_directory: Union[str, Path],
        n_true_bboxes_data: List[List[BboxData]],
        markdowns: List[str],
        codes: List[str],
        classifications_reports_datas: List[ClassificationReportData]
    ):
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok=True, parents=True)
        for model_spec, tag in zip(models_specs, tags):
            if model_spec is None:
                continue
            classification_model_spec_filepath = output_directory / f"{CLASSIFICATION_MODEL_SPEC_PREFIX}_{tag}.pkl"
            with open(classification_model_spec_filepath, 'wb') as out:
                pickle.dump(model_spec, out)
        bboxes_data_filepath = output_directory / BBOXES_DATA_FILENAME
        with open(bboxes_data_filepath, 'wb') as out:
            pickle.dump(n_true_bboxes_data, out)
            
        classification_reports_data_filepath = output_directory / CLASSIFICATIONS_REPORTS_DATAS_FILENAME
        with open(classification_reports_data_filepath, "wb") as out:
            pickle.dump(classifications_reports_datas, out)

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

    def report(
        self,
        models_specs: List[ClassificationModelSpec],
        tags: List[str],
        compare_tag: str,
        output_directory: Union[str, Path],
        n_true_bboxes_data: List[List[BboxData]],
        pseudo_class_names: List[str],
        tops_n: List[int] = [1],
        batch_size: int = 16,
        step_penalty: int = 20
    ) -> List[ClassificationReportData]:
        for model_spec in models_specs:
            if hasattr(model_spec, 'preprocess_input'):
                assert (
                    isinstance(model_spec.preprocess_input, str)
                    or
                    isinstance(model_spec.preprocess_input, Path)
                )
        assert len(models_specs) == len(tags)
        assert compare_tag in tags

        classifications_reports_datas = []
        df_classification_metrics_columns = None
        for model_spec, tag in zip(models_specs, tags):
            logger.info(f"Making inference and counting metrics for '{tag}'...")
            tag_df_classification_metrics = self._inference_classification_and_get_metrics(
                model_spec=model_spec,
                n_true_bboxes_data=n_true_bboxes_data,
                pseudo_class_names=pseudo_class_names,
                tops_n=tops_n,
                batch_size=batch_size,
                step_penalty=step_penalty
            )
            if df_classification_metrics_columns is None:
                df_classification_metrics_columns = tag_df_classification_metrics.columns
            classifications_reports_datas.append(ClassificationReportData(
                df_classification_metrics=tag_df_classification_metrics,
                tops_n=tops_n,
                tag=tag
            ))

        classification_report_data = concat_classifications_reports_datas(
            classifications_reports_datas=classifications_reports_datas,
            df_classification_metrics_columns=df_classification_metrics_columns,
            tops_n=tops_n,
            compare_tag=compare_tag
        )
        markdowns = self._get_markdowns(
            classification_report_data=classification_report_data,
        )
        codes = self._get_codes(tags=tags)
        self._save_report(
            models_specs=models_specs,
            tags=tags,
            output_directory=output_directory,
            n_true_bboxes_data=n_true_bboxes_data,
            markdowns=markdowns,
            codes=codes,
            classifications_reports_datas=classifications_reports_datas
        )
        return classifications_reports_datas

    def report_on_predictions(
        self,
        n_true_bboxes_data: List[List[BboxData]],
        n_pred_bboxes_data: List[List[BboxData]],
        tag: str,
        known_class_names: List[str],
        compare_tag: str,
        output_directory: Union[str, Path],
        pseudo_class_names: List[str],
        tops_n: List[int] = [1],
        step_penalty: int = 20
    ) -> List[ClassificationReportData]:

        logger.info(f"Cunting metrics for '{tag}'...")
        tag_df_classification_metrics = get_df_classification_metrics(
            n_true_bboxes_data=n_true_bboxes_data,
            n_pred_bboxes_data=n_pred_bboxes_data,
            pseudo_class_names=pseudo_class_names,
            known_class_names=known_class_names,
            step_penalty=step_penalty
        )
        df_classification_metrics_columns = tag_df_classification_metrics.columns

        classifications_reports_datas = [ClassificationReportData(
            df_classification_metrics=tag_df_classification_metrics,
            tops_n=tops_n,
            tag=tag
        )]

        classification_report_data = concat_classifications_reports_datas(
            classifications_reports_datas=classifications_reports_datas,
            df_classification_metrics_columns=df_classification_metrics_columns,
            tops_n=tops_n,
            compare_tag=tag
        )
        markdowns = self._get_markdowns(
            classification_report_data=classification_report_data,
        )
        self._save_report(
            models_specs=[None],
            tags=[tag],
            output_directory=output_directory,
            n_true_bboxes_data=n_true_bboxes_data,
            markdowns=markdowns,
            codes=[],
            classifications_reports_datas=classifications_reports_datas
        )

        return classifications_reports_datas

