"""this module contains the logger configuration for the project
will take from other places in the project and centralize the logging configuration right here, including all basic config and all handlers
"""

import logging
import sys
import os
from enum import Enum

import colorlog  # pip install colorlog

from ..utils import find_project_root
from .global_config_mapping import LOGGING_CONFIG

# Create logs directory if it doesn't exist
log_dir = find_project_root() / LOGGING_CONFIG["log_dir"]
os.makedirs(log_dir, exist_ok=True)

# Set up root logger
root_logger = logging.getLogger()
root_logger.setLevel(LOGGING_CONFIG["default_level"])


# Create console handler with colored formatting
console_handler = colorlog.StreamHandler(sys.stdout)

file_handler = logging.FileHandler(os.path.join(log_dir, "debug_graph.log"))
file_handler.setFormatter(logging.Formatter(LOGGING_CONFIG["log_format"]))
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

    def __init__(self, name, level=None, file=None):
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
        # Use centralized defaults if not provided
        if level is None:
            level = LOGGING_CONFIG["default_level"]
        if file is None:
            file = LOGGING_CONFIG["file_logging"]

        super().__init__(name, level)

        if file:
            self.file_handler = logging.FileHandler(
                os.path.join(log_dir, f"{name}.log")
            )
            self.file_handler.setFormatter(
                logging.Formatter(LOGGING_CONFIG["file_format"])
            )
            self.addHandler(self.file_handler)
            self.propagate = False

        self.console = colorlog.StreamHandler(sys.stdout)
        self.console.setFormatter(
            colorlog.ColoredFormatter(
                LOGGING_CONFIG["console_format"],
                log_colors=LOGGING_CONFIG["console_colors"],
            )
        )
        self.addHandler(self.console)
