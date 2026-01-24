"""Tests for chirplab.logging module."""

import logging
from typing import TYPE_CHECKING

from chirplab import logging as chirplab_logging

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger(self) -> None:
        """Test that get_logger returns a Logger instance."""
        logger = chirplab_logging.get_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "chirplab"

    def test_default_level_is_info(self) -> None:
        """Test that the default logging level is INFO."""
        logger = chirplab_logging.get_logger()

        assert logger.level == logging.INFO

    def test_level_info(self) -> None:
        """Test that level='info' sets INFO level."""
        logger = chirplab_logging.get_logger(level="info")

        assert logger.level == logging.INFO

    def test_level_debug(self) -> None:
        """Test that level='debug' sets DEBUG level."""
        logger = chirplab_logging.get_logger(level="debug")

        assert logger.level == logging.DEBUG

    def test_adds_stream_handler(self) -> None:
        """Test that a StreamHandler is added."""
        logger = chirplab_logging.get_logger()
        stream_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.StreamHandler)]

        assert len(stream_handlers) == 5

    def test_file_handler_not_added_by_default(self) -> None:
        """Test that no FileHandler is added when file is None."""
        logger = chirplab_logging.get_logger()
        file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]

        assert len(file_handlers) == 0

    def test_file_handler_added_with_string_path(self, tmp_path: Path) -> None:
        """Test that FileHandler is added when filename path string is provided."""
        log_file = tmp_path / "test.log"
        logger = chirplab_logging.get_logger(filename=str(log_file))
        file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]

        assert len(file_handlers) == 1

    def test_file_handler_added_with_path_object(self, tmp_path: Path) -> None:
        """Test that FileHandler is added when Path object is provided."""
        log_file = tmp_path / "test.log"
        logger = chirplab_logging.get_logger(filename=log_file)
        file_handlers = [handler for handler in logger.handlers if isinstance(handler, logging.FileHandler)]

        assert len(file_handlers) == 2

    def test_logs_written_to_file(self, tmp_path: Path) -> None:
        """Test that log messages are written to the file."""
        log_file = tmp_path / "test.log"
        logger = chirplab_logging.get_logger(level="info", filename=log_file)
        logger.info("Test message")
        for handler in logger.handlers:
            handler.flush()
        content = log_file.read_text()

        assert "Test message" in content

    def test_both_handlers_receive_logs(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that both stream and file handlers receive log messages."""
        log_file = tmp_path / "test.log"
        logger = chirplab_logging.get_logger(level="info", filename=log_file)
        logger.info("Dual output test")
        for handler in logger.handlers:
            handler.flush()
        file_content = log_file.read_text()

        assert "Dual output test" in file_content

        captured = capsys.readouterr()

        assert "Dual output test" in captured.err
