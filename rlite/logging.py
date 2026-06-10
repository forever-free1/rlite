"""Structured logging for rlite training / eval runs."""

import logging
import sys
from pathlib import Path

logger = logging.getLogger("rlite")


def setup_logging(
    level: str = "INFO",
    log_dir: str = "./logs",
    log_file: str | None = None,
) -> None:
    """Configure the rlite root logger with console and optional file output.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
        log_file: Optional specific log file name. Defaults to ``rlite.log``.
    """
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console)

    # file handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fname = log_file or "rlite.log"
        file_handler = logging.FileHandler(Path(log_dir) / fname, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s:%(filename)s:%(lineno)d: %(message)s"
            )
        )
        logger.addHandler(file_handler)

    logger.debug("Logging initialized (level=%s)", level)
