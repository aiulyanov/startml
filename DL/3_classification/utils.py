import logging
import sys


def setup_logger():
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(levelname)s] [%(module)s] %(message)s")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    return logger
