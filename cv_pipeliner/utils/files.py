import tempfile
from pathlib import Path
from typing import Generator

import fsspec
from pathy import Pathy

from cv_pipeliner.logging import logger


def fixed_fsspec_glob(  # this implementation make glob write "gs://" at the start of path
    fs: fsspec.filesystem,
    path: str,
    **kwargs
) -> Generator[str, None, None]:
    path = Pathy(path)
    for some_file in fs.glob(str(path), **kwargs):
        if not some_file.startswith(path.scheme):
            some_file = f"{path.scheme}://{some_file}"
        yield some_file


def copy_files_to_temp_folder(
    directory: str,
    fs: fsspec.filesystem,
    pattern: str = '**'
) -> tempfile.TemporaryDirectory:
    directory = Pathy(directory)
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    for some_file in fixed_fsspec_glob(fs, str(directory / pattern)):
        some_file = Pathy(some_file)
        if fs.isdir(some_file):
            folder_name = Pathy(some_file).name
            (temp_dir_path / folder_name).mkdir(exist_ok=True, parents=True)
        else:
            relative_filepath = some_file.relative_to(directory)
            filepath = temp_dir_path / relative_filepath
            (filepath.parent).mkdir(exist_ok=True, parents=True)
            logger.info(f'copy {some_file} => {filepath}')
            with open(filepath, 'wb') as out:
                with fs.open(some_file, 'rb') as src:
                    out.write(src.read())

    return temp_dir
