"""Tests for datagenerator.log — structured logging setup."""
import logging
import os
import tempfile

from datagenerator.log import setup_logging


class TestSetupLogging:
    def test_creates_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Reset root logger handlers from any prior setup
            root = logging.getLogger()
            root.handlers.clear()

            setup_logging(log_dir=tmpdir, level="INFO")
            logging.getLogger("test").info("hello from test")

            log_path = os.path.join(tmpdir, "geocrawler.log")
            assert os.path.exists(log_path)
            contents = open(log_path).read()
            assert "hello from test" in contents

            # Cleanup
            root.handlers.clear()

    def test_respects_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = logging.getLogger()
            root.handlers.clear()

            setup_logging(log_dir=tmpdir, level="WARNING")
            test_logger = logging.getLogger("test_level")
            test_logger.info("should not appear")
            test_logger.warning("should appear")

            log_path = os.path.join(tmpdir, "geocrawler.log")
            contents = open(log_path).read()
            assert "should not appear" not in contents
            assert "should appear" in contents

            root.handlers.clear()

    def test_no_duplicate_handlers_on_repeated_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = logging.getLogger()
            root.handlers.clear()

            setup_logging(log_dir=tmpdir, level="INFO")
            count_before = len(root.handlers)
            setup_logging(log_dir=tmpdir, level="INFO")
            assert len(root.handlers) == count_before

            root.handlers.clear()

    def test_format_includes_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = logging.getLogger()
            root.handlers.clear()

            setup_logging(log_dir=tmpdir, level="INFO")
            logging.getLogger("mymodule").info("format test")

            log_path = os.path.join(tmpdir, "geocrawler.log")
            contents = open(log_path).read()
            assert "mymodule" in contents
            assert "INFO" in contents
            assert "format test" in contents

            root.handlers.clear()
