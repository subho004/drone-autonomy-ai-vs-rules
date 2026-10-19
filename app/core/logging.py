"""Structured logging utilities for the application.

Provides a standardized plain-text formatter and helpers to configure and
obtain loggers consistently across the codebase.

Follow the project's logging rules: log start/end of important ops and
all failures with stack traces, no prints, and use `logger.exception()`
to attach stack traces when catching.
"""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger with a single stream handler.

    The handler is added only once (idempotent — avoids duplicate log
    lines), but the level is always (re)applied so callers can raise or
    lower verbosity after the first call.
    """

    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``.

    Ensures logging is configured, but never overrides a level already
    set by an explicit `configure_logging(level)` call.
    """

    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
