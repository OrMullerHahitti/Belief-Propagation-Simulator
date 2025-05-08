"""this module contains the logger configuration for the project
will take from other places in the project and centralize the logging configuration right here, including all basic config and all handlers
"""

import logging
import pickle
import sys
import os
from pathlib import Path

import colorlog  # pip install colorlog
import pytest

from utils.path_utils import find_project_root

# Create logs directory if it doesn't exist
log_dir = find_project_root() / "logs"
os.makedirs(log_dir, exist_ok=True)

# Set up root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)


# Create console handler with colored formatting
console_handler = colorlog.StreamHandler(sys.stdout)

file_handler = logging.FileHandler(os.path.join(log_dir, "debug_graph.log"))
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)


class Logger(logging.Logger):
    """
    Custom logger class to add a file handler for logging.
    """

    def __init__(self, name, level=logging.INFO, file=False):
        super().__init__(name, level)

        if file:
            self.file_handler = logging.FileHandler(
                os.path.join(log_dir, f"{name}.log")
            )
            self.file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.addHandler(self.file_handler)
            self.propagate = False

        self.console = colorlog.StreamHandler(sys.stdout)
        self.console.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
        self.addHandler(self.console)
