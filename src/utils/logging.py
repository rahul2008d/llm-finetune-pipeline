"""Structured logging with JSON output for production and pretty-print for local dev.

Provides ``configure_logging`` to set up structlog with correlation IDs,
SageMaker job metadata, and Python warnings integration, plus ``get_logger``
to obtain a bound logger for any module.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
import warnings
from typing import Any

import structlog

_CORRELATION_ID: str = uuid.uuid4().hex


def _add_correlation_id(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject a per-run correlation ID into every log event.

    Args:
        logger: The wrapped logger object.
        method_name: Name of the log method called.
        event_dict: Current event dictionary.

    Returns:
        Enriched event dictionary.
    """
    event_dict["correlation_id"] = _CORRELATION_ID
    return event_dict


def _add_sagemaker_metadata(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add SageMaker training job name if running inside a SageMaker container.

    Args:
        logger: The wrapped logger object.
        method_name: Name of the log method called.
        event_dict: Current event dictionary.

    Returns:
        Enriched event dictionary.
    """
    job_name = os.environ.get("SM_TRAINING_JOB_NAME")
    if job_name:
        event_dict["training_job_name"] = job_name
    return event_dict


def _warnings_to_structlog(
    message: str | Warning,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,
    line: str | None = None,
) -> None:
    """Redirect Python warnings into structlog as structured warning events.

    Args:
        message: Warning message or Warning instance.
        category: Warning category class.
        filename: File that triggered the warning.
        lineno: Line number in the source file.
        file: Unused (kept for warnings API compatibility).
        line: Source line text (unused).
    """
    log = structlog.get_logger("py.warnings")
    log.warning(
        str(message),
        category=category.__name__,
        filename=filename,
        lineno=lineno,
    )


def configure_logging(
    json_output: bool = True,
    log_level: str = "INFO",
) -> None:
    """Configure structlog for the application.

    Call once at startup. Subsequent calls reconfigure in place.

    Args:
        json_output: ``True`` for JSON lines (production), ``False`` for
            coloured pretty-print (local dev).
        log_level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_correlation_id,
        _add_sagemaker_metadata,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Capture Python warnings as structured logs
    warnings.showwarning = _warnings_to_structlog


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module name.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        A structlog ``BoundLogger`` instance.
    """
    return structlog.get_logger(name)


# Keep legacy alias so existing callers don't break
setup_logging = configure_logging

__all__: list[str] = ["configure_logging", "get_logger", "setup_logging"]
