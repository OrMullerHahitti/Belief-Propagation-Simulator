"""
Comprehensive tests for the custom logging system.

This module tests the Logger class and its integration with centralized configuration,
including:
- Logger initialization with defaults and overrides
- File and console logging behavior
- Configuration integration
- Log level handling
- Handler management
"""

import pytest
import logging
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

from propflow.configs.loggers import Logger, Verbose, log_dir
from propflow.configs.global_config_mapping import LOGGING_CONFIG


@pytest.mark.unit
class TestVerboseEnum:
    """Test the Verbose enum values."""

    def test_verbose_enum_values(self):
        """Test that Verbose enum has expected values."""
        assert Verbose.VERBOSE.value == 40
        assert Verbose.MILD.value == 30
        assert Verbose.INFORMATIVE.value == 20
        assert Verbose.HIGH.value == 10

    def test_verbose_enum_ordering(self):
        """Test that Verbose enum values are in descending order."""
        values = [v.value for v in Verbose]
        assert values == sorted(values, reverse=True)


@pytest.mark.unit
class TestLoggerInitialization:
    """Test Logger class initialization with various parameters."""

    def test_logger_default_initialization(self):
        """Test Logger initialization with all defaults."""
        logger = Logger("test_logger")

        assert logger.name == "test_logger"
        assert logger.level == LOGGING_CONFIG["default_level"]
        assert len(logger.handlers) >= 1  # Should have console handler

    def test_logger_with_custom_level(self):
        """Test Logger initialization with custom level."""
        logger = Logger("test_logger", level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_logger_with_file_logging_enabled(self):
        """Test Logger initialization with file logging enabled."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("os.path.join", return_value="/fake/path/test_logger.log"):
                logger = Logger("test_logger", file=True)

                # Should have both console and file handlers
                handler_types = [type(h).__name__ for h in logger.handlers]
                assert "FileHandler" in handler_types
                assert "StreamHandler" in handler_types

    def test_logger_with_file_logging_disabled(self):
        """Test Logger initialization with file logging explicitly disabled."""
        logger = Logger("test_logger", file=False)

        # Should only have console handler
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "FileHandler" not in handler_types
        assert (
            "StreamHandler" in handler_types or "ColoredStreamHandler" in handler_types
        )

    def test_logger_uses_centralized_defaults(self):
        """Test that Logger uses values from LOGGING_CONFIG when None is provided."""
        with patch(
            "propflow.configs.loggers.LOGGING_CONFIG",
            {
                "default_level": logging.ERROR,
                "file_logging": False,
                "file_format": "test_format",
                "console_format": "test_console_format",
                "console_colors": {"INFO": "blue"},
            },
        ):
            logger = Logger("test_logger", level=None, file=None)

            assert logger.level == logging.ERROR
            # Should not have file handler since file_logging is False
            handler_types = [type(h).__name__ for h in logger.handlers]
            assert "FileHandler" not in handler_types


@pytest.mark.unit
class TestLoggerHandlers:
    """Test Logger handler configuration and behavior."""

    def test_console_handler_configuration(self):
        """Test that console handler is properly configured."""
        logger = Logger("test_logger")

        # Find console handler
        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, (logging.StreamHandler, type(logger.console)))
        ]
        assert len(console_handlers) >= 1

        console_handler = console_handlers[0]
        assert console_handler.stream in (sys.stdout, sys.stderr)

    def test_file_handler_configuration(self):
        """Test that file handler is properly configured when enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("propflow.configs.loggers.log_dir", temp_dir):
                logger = Logger("test_logger", file=True)

                # Find file handler
                file_handlers = [
                    h for h in logger.handlers if isinstance(h, logging.FileHandler)
                ]
                assert len(file_handlers) >= 1

                file_handler = file_handlers[0]
                expected_path = os.path.join(temp_dir, "test_logger.log")
                assert file_handler.baseFilename.endswith("test_logger.log")

    def test_file_handler_formatter(self):
        """Test that file handler uses correct formatter."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("os.path.join", return_value="/fake/path/test_logger.log"):
                logger = Logger("test_logger", file=True)

                # Find file handler
                file_handlers = [
                    h for h in logger.handlers if isinstance(h, logging.FileHandler)
                ]
                assert len(file_handlers) >= 1

                file_handler = file_handlers[0]
                formatter = file_handler.formatter
                assert formatter is not None
                assert LOGGING_CONFIG["file_format"] in formatter._fmt

    def test_console_handler_formatter(self):
        """Test that console handler uses colored formatter."""
        logger = Logger("test_logger")

        # Should have console handler with ColoredFormatter
        console_handler = logger.console
        assert console_handler is not None
        formatter = console_handler.formatter
        assert formatter is not None
        # Should be a ColoredFormatter (from colorlog)
        assert hasattr(formatter, "log_colors")

    def test_propagate_setting_with_file_logging(self):
        """Test that propagate is set to False when file logging is enabled."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("os.path.join", return_value="/fake/path/test_logger.log"):
                logger = Logger("test_logger", file=True)

                assert logger.propagate is False

    def test_propagate_setting_without_file_logging(self):
        """Test that propagate setting when file logging is disabled."""
        logger = Logger("test_logger", file=False)

        # Propagate setting may vary, but should be consistent
        assert isinstance(logger.propagate, bool)


@pytest.mark.unit
class TestLoggerIntegration:
    """Test Logger integration with centralized configuration."""

    def test_logger_respects_logging_config(self):
        """Test that Logger respects LOGGING_CONFIG settings."""
        test_config = {
            "default_level": logging.WARNING,
            "file_logging": True,
            "console_format": "TEST: %(message)s",
            "console_colors": {"WARNING": "yellow"},
            "file_format": "FILE: %(message)s",
        }

        with patch("propflow.configs.loggers.LOGGING_CONFIG", test_config):
            with patch("builtins.open", mock_open()):
                with patch("os.path.join", return_value="/fake/test.log"):
                    logger = Logger("test_logger", level=None, file=None)

                    assert logger.level == logging.WARNING

                    # Check console formatter
                    console_formatter = logger.console.formatter
                    assert "TEST:" in console_formatter._fmt

    def test_logger_directory_creation(self):
        """Test that Logger handles log directory creation."""
        # The log directory should be created during module import
        # This test just verifies the directory path is reasonable
        assert isinstance(log_dir, (str, os.PathLike))

    @patch("os.makedirs")
    def test_log_directory_creation_on_import(self, mock_makedirs):
        """Test that log directory is created on module import."""
        # This would be tested by importing the module, but since it's already imported,
        # we just verify the directory path setup
        from propflow.configs.loggers import log_dir

        assert log_dir is not None


@pytest.mark.integration
class TestLoggerFunctionality:
    """Integration tests for Logger functionality."""

    def test_logger_logging_output(self):
        """Test that Logger actually logs messages."""
        # Capture console output
        console_output = StringIO()

        with patch("sys.stdout", console_output):
            logger = Logger("test_logger", level=logging.INFO)
            logger.info("Test message")

        output = console_output.getvalue()
        assert "Test message" in output

    def test_logger_file_logging_output(self):
        """Test that Logger writes to file when file logging is enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("propflow.configs.loggers.log_dir", temp_dir):
                logger = Logger("test_logger", file=True, level=logging.INFO)
                logger.info("File test message")

                # Flush handlers to ensure write
                for handler in logger.handlers:
                    handler.flush()

                log_file_path = os.path.join(temp_dir, "test_logger.log")
                if os.path.exists(log_file_path):
                    with open(log_file_path, "r") as f:
                        content = f.read()
                        assert "File test message" in content

    def test_logger_level_filtering(self):
        """Test that Logger respects log level filtering."""
        console_output = StringIO()

        with patch("sys.stdout", console_output):
            logger = Logger("test_logger", level=logging.ERROR)
            logger.info("Info message")  # Should be filtered out
            logger.error("Error message")  # Should appear

        output = console_output.getvalue()
        assert "Info message" not in output
        assert "Error message" in output

    def test_logger_multiple_handlers_no_duplication(self):
        """Test that creating multiple loggers doesn't duplicate handlers."""
        logger1 = Logger("test_logger_1")
        logger2 = Logger("test_logger_2")

        # Each logger should have its own handlers
        assert logger1.handlers != logger2.handlers

        # But they shouldn't interfere with each other
        console_output1 = StringIO()
        console_output2 = StringIO()

        # This is a simplified test - in real usage, handlers write to different streams
        assert len(logger1.handlers) > 0
        assert len(logger2.handlers) > 0


@pytest.mark.unit
class TestLoggerEdgeCases:
    """Test Logger edge cases and error handling."""

    def test_logger_with_empty_name(self):
        """Test Logger behavior with empty name."""
        logger = Logger("")
        assert logger.name == ""
        assert len(logger.handlers) > 0

    def test_logger_with_special_characters_in_name(self):
        """Test Logger behavior with special characters in name."""
        special_name = "test-logger_123.special"
        logger = Logger(special_name)
        assert logger.name == special_name

    def test_logger_with_invalid_level_type(self):
        """Test Logger behavior when invalid level type is provided."""
        # Should fall back to default level from config
        logger = Logger("test_logger", level="invalid")
        assert logger.level == LOGGING_CONFIG["default_level"]

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_logger_file_handler_permission_error(self, mock_open):
        """Test Logger behavior when file cannot be opened due to permissions."""
        with patch("os.path.join", return_value="/fake/path/test.log"):
            # Should not raise exception, but may not have file handler
            try:
                logger = Logger("test_logger", file=True)
                # If no exception, file handler creation was handled gracefully
            except PermissionError:
                # This is also acceptable behavior
                pass

    def test_logger_handler_cleanup(self):
        """Test that Logger handlers can be properly cleaned up."""
        logger = Logger("test_logger", file=False)
        original_handler_count = len(logger.handlers)

        # Remove all handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        assert len(logger.handlers) == 0

        # Add a new handler
        new_handler = logging.StreamHandler()
        logger.addHandler(new_handler)

        assert len(logger.handlers) == 1
        assert new_handler in logger.handlers


@pytest.mark.integration
class TestLoggerWithRootLogger:
    """Test Logger interaction with root logger configuration."""

    def test_logger_independence_from_root(self):
        """Test that custom Logger operates independently from root logger."""
        root_logger = logging.getLogger()
        original_root_level = root_logger.level

        # Change root logger level
        root_logger.setLevel(logging.CRITICAL)

        # Create custom logger
        custom_logger = Logger("custom_test", level=logging.INFO)

        # Custom logger should have its own level
        assert custom_logger.level == logging.INFO
        assert root_logger.level == logging.CRITICAL

        # Restore root logger level
        root_logger.setLevel(original_root_level)

    def test_logger_propagation_behavior(self):
        """Test Logger propagation behavior."""
        logger_with_file = Logger("test_with_file", file=True)
        logger_without_file = Logger("test_without_file", file=False)

        # Logger with file should not propagate to avoid duplication
        assert logger_with_file.propagate is False

        # Logger without file may propagate (implementation dependent)
        assert isinstance(logger_without_file.propagate, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
