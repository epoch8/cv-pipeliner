import logging
import tarfile

from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm


logger = logging.getLogger(__name__)


def download_and_extract_tar_gz_to_directory(
    url: str,
    directory: Union[str, Path]
):
    directory = Path()
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / 'tempfile.tar.gz'

    logger.info(f"Downloading tar.gz archive from {url} ...")
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(filepath, 'wb') as file, tqdm(
            desc=str(filepath),
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as tbar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            tbar.update(size)

    logger.info(f"Extracting all to {directory}...")
    tar = tarfile.open(filepath)
    tar.extractall(path=directory)
    tar.close()

    filepath.unlink()
    logger.info(f"Saved to {directory}")
