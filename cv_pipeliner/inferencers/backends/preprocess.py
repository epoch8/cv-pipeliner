import importlib
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Union

import fsspec
import numpy as np


def get_preprocess_input_from_script_file(script_file: Union[str, Path]) -> Callable[[List[np.ndarray]], np.ndarray]:
    with fsspec.open(script_file, "r") as src:
        script_code = src.read()
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        module_folder = tmpdirname / "module"
        module_folder.mkdir()
        script_file = module_folder / f"preprocess_input_{tmpdirname.name}.py"
        with open(script_file, "w") as out:
            out.write(script_code)
        sys.path.append(str(script_file.parent.absolute()))
        module = importlib.import_module(script_file.stem)
        importlib.reload(module)
        sys.path.pop()
    return module.preprocess_input
