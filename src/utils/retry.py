"""Retry decorator with exponential backoff and structured logging.

Provides ``with_retry`` — a configurable decorator that retries on specified
exception types and logs every attempt via structlog.  The legacy
``retry_with_backoff`` alias is preserved for backward compatibility.
"""

from collections.abc import Callable
from typing import Any, TypeVar

import structlog
from botocore.exceptions import BotoCoreError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _log_retry(retry_state: RetryCallState) -> None:
    """Emit a structured warning on each retry attempt.

    Args:
        retry_state: Current retry state from tenacity.
    """
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying",
        attempt=retry_state.attempt_number,
        fn=getattr(retry_state.fn, "__name__", str(retry_state.fn)),
        exception_type=type(exception).__name__ if exception else None,
        exception_message=str(exception) if exception else None,
    )


def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2,
    retryable_exceptions: tuple[type[Exception], ...] = (BotoCoreError,),
) -> Callable[[F], F]:
    """Create a retry decorator with exponential backoff.

    Args:
        max_attempts: Total number of attempts (including the first call).
        backoff_factor: Multiplier for exponential wait (``factor ** attempt``).
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        A decorator that wraps the target function with retry logic.
    """
    return retry(  # type: ignore[return-value]
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff_factor, min=1, max=60),
        retry=retry_if_exception_type(retryable_exceptions),
        before_sleep=_log_retry,
        reraise=True,
    )


# Legacy alias kept for backward compatibility
retry_with_backoff = with_retry

__all__: list[str] = ["with_retry", "retry_with_backoff"]
