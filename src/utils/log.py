"""
Central structlog configuration for the BAPRO pipeline.

Usage in any module:
    from src.utils.log import get_logger
    log = get_logger(__name__)

Environment variables:
    LOG_LEVEL   — DEBUG / INFO / WARNING / ERROR  (default: INFO)
    LOG_FORMAT  — console / json                  (default: console)

Long-running task convention:
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    log.info("start... <task_name>", ts=now)
    ...
    log.info("finish... <task_name>", ts=datetime.now().strftime("%Y/%m/%d %H:%M"), n=count)
"""
import logging
import os
import sys

import structlog

_configured = False


def _setup() -> None:
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv("LOG_FORMAT", "console")

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y/%m/%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        # colors=False keeps output ASCII-safe for Windows cp1252 terminals
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structlog logger bound with the given module name."""
    _setup()
    return structlog.get_logger(name)
