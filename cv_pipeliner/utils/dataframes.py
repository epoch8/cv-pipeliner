from typing import List

import numpy as np
import pandas as pd


def transpose_columns_and_write_diffs_to_df_with_tags(
    df_with_tags: pd.DataFrame,  # columns are like f'{column_name} [{tag}]'
    columns: List[str],
    tags: List[str],
    compare_tag: str,
    use_colors: bool = True,
):
    assert compare_tag in tags

    transposed_columns = []
    for column in columns:
        for tag in tags:
            transposed_columns.append(f"{column} [{tag}]")

    transposed_df_with_tags = df_with_tags[transposed_columns]

    def round_nan(value, ndigits):
        if not np.isnan(value):
            return round(value, ndigits)
        else:
            return value

    def float_to_int_if_needed(value):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    for column in columns:
        transposed_df_with_tags.loc[:, f"{column} [{compare_tag}]"] = [
            round_nan(value, 3) for value in transposed_df_with_tags[f"{column} [{compare_tag}]"]
        ]
        compare_tag_values = transposed_df_with_tags[f"{column} [{compare_tag}]"]
        for tag in tags:
            if tag == compare_tag:
                continue
            tag_values = transposed_df_with_tags[f"{column} [{tag}]"]
            tag_diffs = (tag_values.astype(float) - compare_tag_values.astype(float)) * np.reciprocal(
                compare_tag_values.astype(float)
            )

            tag_values = [round_nan(tag_value, 3) for tag_value in tag_values]
            tag_signs = ["+" if tag_diff > 0 else "" for tag_diff in tag_diffs]
            tag_suffixes = [
                f"({tag_sign}{int(round(100 * tag_diff))}%)"
                if (not np.isnan(tag_diff) and not np.isinf(tag_diff) and np.abs(tag_diff) > 0.01)
                else ""
                for tag_sign, tag_diff in zip(tag_signs, tag_diffs)
            ]
            if use_colors:
                tag_colors = ["green" if tag_sign == "+" else "red" for tag_sign in tag_signs]
                tag_suffixes = [
                    f"<font color='{tag_color}'>{tag_suffix}</font>"
                    for tag_color, tag_suffix in zip(tag_colors, tag_suffixes)
                ]

            transposed_df_with_tags.loc[:, f"{column} [{tag}]"] = [
                f"{float_to_int_if_needed(tag_value)} {tag_suffix}"
                for tag_value, tag_suffix in zip(tag_values, tag_suffixes)
            ]

    return transposed_df_with_tags
