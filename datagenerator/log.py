"""Centralised logging setup for geocrawler-datagenerator.

Usage:
    from datagenerator.log import setup_logging
    setup_logging()  # call once in main.py

    # In any module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("message")
"""
import logging
import os
import tempfile


def setup_logging(log_dir=None, level=None):
    """Configure root logger with file + console handlers.

    Args:
        log_dir: Directory for log file. Falls back to GC_LOG_DIR env var,
                 then the system temporary directory.
        level: Log level string. Falls back to GC_LOG_LEVEL env var, then INFO.
    """
    if log_dir is None:
        log_dir = os.environ.get("GC_LOG_DIR", tempfile.gettempdir())
    if level is None:
        level = os.environ.get("GC_LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "geocrawler.log")
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
