import logging
import sys
import os
from pathlib import Path
from datetime import datetime

from tensorflow import get_logger as tf_get_logger

LOGS_DIRECTORY = Path(__file__).parent / '__logs__'

logger = logging.getLogger('cv-pipeliner')

if os.environ.get('CV_PIPELINER_LOGGING', False):
    LOGS_DIRECTORY.mkdir(exist_ok=True, parents=True)
    LOG_FILENAME = datetime.now().strftime("%Y-%m-%d_%Hh.logs")

    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
    )

    file_handler = logging.FileHandler(LOGS_DIRECTORY / LOG_FILENAME,
                                    encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.propagate = False

    tf_logger = tf_get_logger()
    tf_logger.addHandler(file_handler)
    tf_logger.propagate = False
    for handler in tf_logger.handlers:
        handler.setFormatter(formatter)
