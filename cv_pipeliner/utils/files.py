import tempfile
from pathlib import Path

import fsspec
from pathy import Pathy

from cv_pipeliner.logging import logger


def copy_files_from_directory_to_temp_directory(directory: str) -> tempfile.TemporaryDirectory:
    directory_openfile = fsspec.open(directory)
    directory = Pathy(directory)
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    for some_file in fsspec.open_files(str(directory / '**'), 'rb'):
        some_file_path = Pathy(some_file.path)
        relative_filepath = some_file_path.relative_to(directory_openfile.path)
        filepath = temp_dir_path / relative_filepath
        (filepath.parent).mkdir(exist_ok=True, parents=True)
        logger.info(f'copy {some_file_path} => {filepath}')
        with open(filepath, 'wb') as out:
            with some_file as src:
                out.write(src.read())

    return temp_dir
