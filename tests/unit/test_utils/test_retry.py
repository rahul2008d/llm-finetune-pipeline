"""Unit tests for utils.retry module."""

import pytest
from botocore.exceptions import BotoCoreError

from src.utils.retry import with_retry


class TestWithRetry:
    """Tests for the with_retry decorator."""

    def test_succeeds_immediately(self) -> None:
        """Test that successful calls are not retried."""
        call_count = 0

        @with_retry(max_attempts=3, backoff_factor=0.01, retryable_exceptions=(ValueError,))
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = succeed()
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_matching_exception(self) -> None:
        """Test that matching exceptions trigger retries."""
        call_count = 0

        @with_retry(max_attempts=3, backoff_factor=0.01, retryable_exceptions=(ValueError,))
        def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "ok"

        result = fail_then_succeed()
        assert result == "ok"
        assert call_count == 3

    def test_does_not_retry_non_matching_exception(self) -> None:
        """Test that non-matching exceptions propagate immediately."""

        @with_retry(max_attempts=3, backoff_factor=0.01, retryable_exceptions=(ValueError,))
        def raise_type_error() -> None:
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            raise_type_error()

    def test_exhausts_retries(self) -> None:
        """Test that exhausted retries re-raise the exception."""

        @with_retry(max_attempts=2, backoff_factor=0.01, retryable_exceptions=(RuntimeError,))
        def always_fail() -> None:
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            always_fail()

    def test_default_retries_botocore_error(self) -> None:
        """Test that the default retryable exception is BotoCoreError."""
        call_count = 0

        @with_retry(max_attempts=2, backoff_factor=0.01)
        def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise BotoCoreError()
            return "recovered"

        assert fail_once() == "recovered"
        assert call_count == 2

    def test_non_retryable_exception_raises_immediately(self) -> None:
        """Non-retryable exceptions are not retried."""
        @with_retry(max_attempts=3, backoff_factor=0.01)
        def always_fail() -> None:
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            always_fail()
