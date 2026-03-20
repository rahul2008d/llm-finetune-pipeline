"""Tests for SageMaker inference handler."""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.inference import input_fn, output_fn, predict_fn


# ── input_fn Tests ──────────────────────────────────────────────


class TestInputFn:
    """Tests for input_fn parsing and validation."""

    def test_input_fn_json(self) -> None:
        """Verify JSON parsing with all fields."""
        payload = json.dumps(
            {
                "prompt": "Hello, world!",
                "max_new_tokens": 128,
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 40,
                "do_sample": True,
                "repetition_penalty": 1.2,
                "stop_sequences": ["###"],
            }
        )

        result = input_fn(payload, "application/json")

        assert result["prompt"] == "Hello, world!"
        assert result["max_new_tokens"] == 128
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.8
        assert result["top_k"] == 40
        assert result["do_sample"] is True
        assert result["repetition_penalty"] == 1.2
        assert result["stop_sequences"] == ["###"]

    def test_input_fn_defaults(self) -> None:
        """Verify defaults applied for missing fields."""
        payload = json.dumps({"prompt": "Test prompt"})

        result = input_fn(payload, "application/json")

        assert result["prompt"] == "Test prompt"
        assert result["max_new_tokens"] == 256
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["top_k"] == 50
        assert result["do_sample"] is True
        assert result["repetition_penalty"] == 1.1

    def test_input_fn_text_plain(self) -> None:
        """Verify plain text creates prompt-only input."""
        result = input_fn("Just a plain text prompt", "text/plain")

        assert result["prompt"] == "Just a plain text prompt"
        assert result["max_new_tokens"] == 256  # default applied

    def test_input_fn_invalid_json(self) -> None:
        """Verify error handling for invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            input_fn("{invalid json", "application/json")

    def test_input_fn_missing_prompt(self) -> None:
        """Verify error when prompt field is missing."""
        with pytest.raises(ValueError, match="Missing required field"):
            input_fn(json.dumps({"temperature": 0.5}), "application/json")

    def test_input_fn_unsupported_content_type(self) -> None:
        """Verify error for unsupported content type."""
        with pytest.raises(ValueError, match="Unsupported content type"):
            input_fn("data", "text/xml")

    def test_input_fn_bounds_checking(self) -> None:
        """Verify parameter bounds are enforced."""
        payload = json.dumps(
            {
                "prompt": "Test",
                "temperature": 10.0,  # exceeds max of 2.0
                "max_new_tokens": -5,  # below min of 1
            }
        )

        result = input_fn(payload, "application/json")

        assert result["temperature"] == 2.0
        assert result["max_new_tokens"] == 1

    def test_input_fn_string_json(self) -> None:
        """Verify bare string JSON is handled as prompt."""
        payload = json.dumps("A simple prompt string")
        result = input_fn(payload, "application/json")
        assert result["prompt"] == "A simple prompt string"


# ── predict_fn Tests ────────────────────────────────────────────


class TestPredictFn:
    """Tests for predict_fn generation."""

    def test_predict_fn_returns_expected_keys(self) -> None:
        """Mock model, verify output dict has all required keys."""
        # Mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters.return_value = iter([mock_param])

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        # Setup input_ids shape
        input_ids = MagicMock()
        input_ids.shape = [1, 5]
        input_ids.to.return_value = input_ids
        mock_tokenizer.return_value["input_ids"] = input_ids
        mock_tokenizer.return_value["attention_mask"] = MagicMock()
        mock_tokenizer.return_value["attention_mask"].to.return_value = (
            mock_tokenizer.return_value["attention_mask"]
        )

        # Mock generate output — 5 input tokens + 3 generated tokens
        import torch

        output_ids = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])
        mock_model.generate.return_value = output_ids
        mock_tokenizer.decode.return_value = "Generated response text"

        input_data = {
            "prompt": "Test prompt",
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "stop_sequences": [],
        }

        with patch("src.serving.inference.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            # Use real torch for tensor ops
            import src.serving.inference as inf_mod

            # Temporarily restore real torch for the function execution
            original_torch = inf_mod.torch
            inf_mod.torch = torch
            try:
                result = predict_fn(input_data, (mock_model, mock_tokenizer))
            finally:
                inf_mod.torch = original_torch

        assert "generated_text" in result
        assert "num_input_tokens" in result
        assert "num_output_tokens" in result
        assert "latency_ms" in result
        assert "finish_reason" in result
        assert isinstance(result["latency_ms"], float)


# ── output_fn Tests ─────────────────────────────────────────────


class TestOutputFn:
    """Tests for output_fn serialization."""

    def test_output_fn_json(self) -> None:
        """Verify JSON serialization."""
        prediction = {
            "generated_text": "Hello!",
            "num_input_tokens": 5,
            "num_output_tokens": 3,
            "latency_ms": 42.5,
            "finish_reason": "stop",
        }

        result = output_fn(prediction, "application/json")
        parsed = json.loads(result)

        assert parsed["generated_text"] == "Hello!"
        assert parsed["num_input_tokens"] == 5
        assert parsed["latency_ms"] == 42.5

    def test_output_fn_includes_request_id(self) -> None:
        """Verify UUID present when not provided."""
        prediction = {"generated_text": "Test"}

        result = output_fn(prediction, "application/json")
        parsed = json.loads(result)

        assert "request_id" in parsed
        # Verify it's a valid UUID
        uuid.UUID(parsed["request_id"])

    def test_output_fn_preserves_existing_request_id(self) -> None:
        """Verify existing request_id is not overwritten."""
        existing_id = "custom-request-id-123"
        prediction = {"generated_text": "Test", "request_id": existing_id}

        result = output_fn(prediction, "application/json")
        parsed = json.loads(result)

        assert parsed["request_id"] == existing_id

    def test_output_fn_unsupported_accept_type(self) -> None:
        """Verify error for unsupported accept type."""
        with pytest.raises(ValueError, match="Unsupported accept type"):
            output_fn({"generated_text": "Test"}, "text/xml")
