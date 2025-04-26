"""this module contains the logger configuration for the project
will take from other places in the project and centralize the logging configuration right here, including all basic config and all handlers"""
import logging
import pickle
import sys
import os
from pathlib import Path

import colorlog  # pip install colorlog
import pytest

# Create logs directory if it doesn't exist
log_dir = 'test_logs'
os.makedirs(log_dir, exist_ok=True)

# Set up root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear any existing handlers
if root_logger.handlers:
    root_logger.handlers.clear()

# Create console handler with colored formatting
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

file_handler = logging.FileHandler(os.path.join(log_dir, 'debug_graph.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(file_handler)

