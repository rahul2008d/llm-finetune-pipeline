"""Root-level test configuration and shared fixtures."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure a clean environment for each test."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test artifacts.

    Args:
        tmp_path: Pytest-provided temporary path.

    Returns:
        Path to the temporary directory.
    """
    return tmp_path


@pytest.fixture()
def sample_dataset_dict() -> list[dict[str, str]]:
    """Provide sample dataset rows for testing.

    Returns:
        List of sample instruction/input/output dictionaries.
    """
    return [
        {
            "instruction": "Summarize the text.",
            "input": "Machine learning is a subset of AI.",
            "output": "ML is a branch of artificial intelligence.",
        },
        {
            "instruction": "Translate to French.",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous?",
        },
        {
            "instruction": "Explain the concept.",
            "input": "What is gradient descent?",
            "output": "An optimization algorithm that iteratively adjusts parameters.",
        },
    ]


@pytest.fixture()
def mock_boto3_client() -> MagicMock:
    """Provide a mock boto3 client.

    Returns:
        MagicMock configured as a boto3 client.
    """
    return MagicMock()


@pytest.fixture()
def sample_yaml_config(tmp_path: Path) -> Path:
    """Create a sample YAML config file for testing.

    Args:
        tmp_path: Pytest-provided temporary path.

    Returns:
        Path to the created YAML config file.
    """
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        """model_id: test-model
dataset: test-dataset
output_dir: /tmp/test-output
epochs: 1
batch_size: 2
learning_rate: 0.001
lora_rank: 16
lora_alpha: 32
use_qlora: true
use_dora: false
"""
    )
    return config_path
