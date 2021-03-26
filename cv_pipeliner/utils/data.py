import json
from pathlib import Path
from typing import Union, Dict
from collections import defaultdict

import fsspec


def get_label_to_description(
    label_to_description_dict: Union[str, Path, Dict, None],
    default_description: str = 'No description.'
) -> Dict[str, str]:
    if label_to_description_dict is None:
        label_to_description_dict = {}
    elif isinstance(label_to_description_dict, str) or isinstance(label_to_description_dict, Path):
        with fsspec.open(label_to_description_dict, 'r') as src:
            label_to_description_dict = json.load(src)

    label_to_description = defaultdict(lambda: default_description)
    label_to_description['unknown'] = default_description
    for k in label_to_description_dict:
        label_to_description[k] = label_to_description_dict[k]

    return label_to_description
