"""Centralized Logger Configuration for the PropFlow Project.

This module sets up a standardized logging system for the entire application.
It uses the `colorlog` library to provide colored console output for improved
readability and also supports logging to files.

The configuration is driven by the `LOGGING_CONFIG` dictionary in the
`global_config_mapping` module. It initializes a root logger and provides a
custom `Logger` class that can be instantiated anywhere in the project for
consistent logging behavior.
"""

import logging
import sys
import os
from enum import Enum

import colorlog

from ..utils.path_utils import find_project_root
from .global_config_mapping import LOGGING_CONFIG

# Ensure the log directory exists.
log_dir = find_project_root() / LOGGING_CONFIG["log_dir"]
os.makedirs(log_dir, exist_ok=True)

# Configure the root logger.
root_logger = logging.getLogger()
root_logger.setLevel(LOGGING_CONFIG["default_level"])

# Add a default file handler to the root logger for general debug logs.
file_handler = logging.FileHandler(os.path.join(log_dir, "debug_graph.log"))
file_handler.setFormatter(logging.Formatter(LOGGING_CONFIG["log_format"]))
root_logger.addHandler(file_handler)


class Verbose(Enum):
    """Represents different verbosity levels for logging."""
    VERBOSE = 40
    MILD = 30
    INFORMATIVE = 20
    HIGH = 10


class Logger(logging.Logger):
    """A custom logger that provides standardized console and file logging.

    This class extends the standard `logging.Logger` to automatically configure
    a colored console handler and an optional file handler based on the global
    logging configuration.

    Attributes:
        file_handler (logging.FileHandler): The handler for logging to a file,
            which is only created if file logging is enabled.
        console (colorlog.StreamHandler): The handler for logging to the console
            with colored output.
    """

    def __init__(self, name: str, level: int = None, file: bool = None):
        """Initializes the custom logger.

        Args:
            name: The name of the logger, typically `__name__`.
            level: The logging level. If not provided, it defaults to the
                `default_level` in `LOGGING_CONFIG`.
            file: Whether to enable file-based logging for this logger instance.
                If not provided, it defaults to `file_logging` in `LOGGING_CONFIG`.
        """
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
            self.propagate = False  # Prevent logs from reaching the root logger's handlers

        self.console = colorlog.StreamHandler(sys.stdout)
        self.console.setFormatter(
            colorlog.ColoredFormatter(
                LOGGING_CONFIG["console_format"],
                log_colors=LOGGING_CONFIG["console_colors"],
            )
        )
        self.addHandler(self.console)
