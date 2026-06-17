from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from cv_pipeliner.inferencers.backends.preprocess import get_preprocess_input_from_script_file


def resolve_preprocess_input(preprocess_input: Optional[Union[str, Path, Callable]]) -> Callable:
    if preprocess_input is None:
        return lambda input: np.array(input)
    if isinstance(preprocess_input, (str, Path)):
        return get_preprocess_input_from_script_file(preprocess_input)
    return preprocess_input


def get_tflite_input_details(interpreter: Any) -> Tuple[int, Tuple[int, ...]]:
    input_details = interpreter.get_input_details()
    return input_details[0]["index"], tuple(input_details[0]["shape"])


def get_tflite_output_details(interpreter: Any) -> Tuple[int, Tuple[int, ...]]:
    output_details = interpreter.get_output_details()
    return output_details[0]["index"], tuple(output_details[0]["shape"])
