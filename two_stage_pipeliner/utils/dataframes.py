from typing import List

import numpy as np
import pandas as pd


def transpose_columns_and_write_diffs_to_df_with_tags(
    df_with_tags: pd.DataFrame,  # columns are like f'{column_name} [{tag}]'
    columns: List[str],
    tags: List[str],
    compare_tag: str,
    inplace: bool = False
):
    assert compare_tag in tags

    transposed_columns = []
    for column in columns:
        for tag in tags:
            transposed_columns.append(f'{column} [{tag}]')

    transposed_df_with_tags = df_with_tags[transposed_columns]

    def round_nan(value, ndigits):
        if value == value:
            return round(value, ndigits)
        else:
            return value

    for column in columns:
        df_with_tags[f'{column} [{compare_tag}]'] = [
            round_nan(value, 3)
            for value in transposed_df_with_tags[f'{column} [{compare_tag}]']
        ]
        compare_tag_values = df_with_tags[f'{column} [{compare_tag}]']
        for tag in tags:
            if tag == compare_tag:
                continue
            tag_values = transposed_df_with_tags[f'{column} [{tag}]']
            tag_diffs = (tag_values - compare_tag_values) * np.reciprocal(compare_tag_values)

            tag_values = [round_nan(tag_value, 3) for tag_value in tag_values]
            tag_signs = ['+' if tag_diff > 0 else '' for tag_diff in tag_diffs]
            tag_suffixes = [
                f"({tag_sign}{int(round(100 * tag_diff))}%)"
                if (tag_diff == tag_diff) and (not np.isinf(tag_diff) and np.abs(tag_diff) > 1e-6) else ''
                for tag_sign, tag_diff in zip(tag_signs, tag_diffs)
            ]

            transposed_df_with_tags[f'{column} [{tag}]'] = [
                f"{tag_value} {tag_suffix}"
                for tag_value, tag_suffix in zip(tag_values, tag_suffixes)
            ]

    return transposed_df_with_tags
