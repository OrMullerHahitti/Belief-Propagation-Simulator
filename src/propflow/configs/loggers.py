"""this module contains the logger configuration for the project
will take from other places in the project and centralize the logging configuration right here, including all basic config and all handlers
"""

import logging
import sys
import os
from enum import Enum

import colorlog  # pip install colorlog

from src.propflow.utils import find_project_root

# Create logs directory if it doesn't exist
log_dir = find_project_root() / "configs" / "logs"
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

class Verbose(Enum):
    """
    Enum representing different verbosity levels for logging.
    """
    VERBOSE = 40
    MILD = 30
    INFORMATIVE = 20
    HIGH = 10


class Logger(logging.Logger):
    """
    Custom logger class to add a file handler for logging.
    """

    def __init__(self, name, level=logging.INFO, file=False):
        """
        Initializes a custom logging handler that supports both file and console-based logging
        with colored output for better readability. The logger can log to a file if specified
        and always outputs colored logs to the console.

        Attributes:
            file_handler (logging.FileHandler): Handles logging to a file. Initialized if file
                logging is enabled.
            console (colorlog.StreamHandler): Handles logging to the console with colored
                messages.

        Args:
            name (str): The name of the logger.
            level (int): The logging level. Default is logging.INFO.
            file (bool): Whether to enable file-based logging. Default is False.
        """
        super().__init__(name, level)

        if file:
            self.file_handler = logging.FileHandler(
                os.path.join(log_dir, f"{name}.log")
            )
            self.file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s  - %(message)s"
                )
            )
            self.addHandler(self.file_handler)
            self.propagate = False

        self.console = colorlog.StreamHandler(sys.stdout)
        self.console.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(message)s",
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
