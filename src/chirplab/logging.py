"""Module for logging."""

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

type LevelType = Literal["info", "debug"]


def get_logger(
    level: LevelType = "info",
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename: None | str | Path = None,
) -> logging.Logger:
    """
    Get a logger.

    Parameters
    ----------
    level
        Logging level.
    fmt
        Logging format.
    filename
        File path to write logs to.

    Returns
    -------
    logger
        Logger instance.
    """
    logger = logging.getLogger("chirplab")

    if level.lower() == "info":
        logger.setLevel(logging.INFO)
    elif level.lower() == "debug":
        logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
