"""Unit tests for utils.logging module."""

import json
import logging
import os
import warnings

import pytest
import structlog

from src.utils.logging import configure_logging, get_logger


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that JSON mode produces parseable JSON on stdout."""
        configure_logging(json_output=True, log_level="DEBUG")
        log = get_logger("test.json")
        log.info("hello", key="value")

        captured = capsys.readouterr()
        line = captured.out.strip().split("\n")[-1]
        data = json.loads(line)
        assert data["event"] == "hello"
        assert data["key"] == "value"
        assert "timestamp" in data
        assert "correlation_id" in data

    def test_pretty_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that non-JSON mode produces human-readable output."""
        configure_logging(json_output=False, log_level="INFO")
        log = get_logger("test.pretty")
        log.info("pretty message")

        captured = capsys.readouterr()
        assert "pretty message" in captured.out

    def test_log_level_filtering(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that messages below the configured level are suppressed."""
        configure_logging(json_output=True, log_level="WARNING")
        log = get_logger("test.level")
        log.info("should not appear")
        log.warning("should appear")

        captured = capsys.readouterr()
        assert "should not appear" not in captured.out
        assert "should appear" in captured.out

    def test_correlation_id_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that every log line includes correlation_id."""
        configure_logging(json_output=True, log_level="DEBUG")
        log = get_logger("test.corr")
        log.info("check id")

        captured = capsys.readouterr()
        line = captured.out.strip().split("\n")[-1]
        data = json.loads(line)
        assert "correlation_id" in data
        assert len(data["correlation_id"]) == 32  # hex UUID

    def test_sagemaker_job_name(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that SM_TRAINING_JOB_NAME is injected when present."""
        monkeypatch.setenv("SM_TRAINING_JOB_NAME", "my-training-job")
        configure_logging(json_output=True, log_level="DEBUG")
        log = get_logger("test.sm")
        log.info("sm event")

        captured = capsys.readouterr()
        line = captured.out.strip().split("\n")[-1]
        data = json.loads(line)
        assert data.get("training_job_name") == "my-training-job"

    def test_warnings_captured(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that Python warnings are redirected to structlog."""
        configure_logging(json_output=True, log_level="DEBUG")
        warnings.warn("test deprecation", DeprecationWarning, stacklevel=1)

        captured = capsys.readouterr()
        assert "test deprecation" in captured.out


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_bound_logger(self) -> None:
        """Test that get_logger returns a structlog BoundLogger."""
        configure_logging(json_output=True)
        log = get_logger("mymodule")
        # structlog loggers are proxy objects; check they have .info
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
