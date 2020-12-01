import json
from pathlib import Path
from typing import Union, Dict, Callable

import fsspec


def get_label_to_description(
    label_to_description_dict: Union[str, Path, Dict]
) -> Callable[[str], str]:
    if isinstance(label_to_description_dict, str) or isinstance(label_to_description_dict, Path):
        with fsspec.open(label_to_description_dict, 'r') as src:
            label_to_description_dict = json.load(src)

    label_to_description_dict['unknown'] = 'No description.'

    def label_to_description(label: str) -> str:
        if label in label_to_description_dict:
            return label_to_description_dict[label]
        else:
            return label_to_description_dict['unknown']

    return label_to_description
