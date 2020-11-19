import tempfile
from pathlib import Path

import fsspec
from pathy import Pathy
from typing import List

from cv_pipeliner.logging import logger


# this implementation make glob write schemes like "gs://" at the start of path
def fsspec_glob(urlpath: str, mode: str = 'rb', **kwargs) -> List[str]:
    scheme = Pathy(urlpath).scheme
    files = fsspec.open_files(urlpath=urlpath, mode=mode, **kwargs)
    paths = []
    for f in files:
        if Pathy(f.path).scheme != scheme:
            paths.append(f"{scheme}://{f.path}")
        else:
            paths.append(f.path)
    return paths


def copy_files_to_temp_folder(
    directory: str,
    pattern: str = '**'
) -> tempfile.TemporaryDirectory:
    directory = Pathy(directory)
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)

    fs = fsspec.filesystem(Pathy(directory).scheme)
    for some_file in fsspec.open_files(str(directory / pattern), 'rb'):
        some_file_path = Pathy(some_file.path)
        if some_file_path.scheme != directory.scheme:  # happens in GCSFS.glob
            some_file_path = Pathy(f"{directory.scheme}://{some_file_path}")
        if fs.isdir(str(some_file_path)):
            folder_name = Pathy(some_file_path).name
            (temp_dir_path / folder_name).mkdir(exist_ok=True, parents=True)
        else:
            relative_filepath = some_file_path.relative_to(directory)
            filepath = temp_dir_path / relative_filepath
            (filepath.parent).mkdir(exist_ok=True, parents=True)
            logger.info(f'copy {some_file_path} => {filepath}')
            with open(filepath, 'wb') as out:
                with some_file as src:
                    out.write(src.read())

    return temp_dir
