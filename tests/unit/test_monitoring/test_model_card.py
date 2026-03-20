"""Unit tests for monitoring.model_card module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from src.config.training import (
    LoRAConfig,
    ModelConfig,
    QuantizationConfig,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from src.monitoring.model_card import ModelCardGenerator


@pytest.fixture()
def sample_config() -> TrainingJobConfig:
    """Create a sample TrainingJobConfig for testing."""
    return TrainingJobConfig(
        experiment_name="test-experiment",
        run_name="test-run",
        model=ModelConfig(model_name_or_path="meta-llama/Llama-3-8B"),
        quantization=QuantizationConfig(method="qlora"),
        lora=LoRAConfig(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj"]),
        training=TrainingHyperparameters(),
        dataset_path="/tmp/dataset",
        dataset_id="dataset-v1",
    )


@pytest.fixture()
def sample_result() -> MagicMock:
    """Create a mock TrainingResult."""
    result = MagicMock()
    result.run_id = "run-123"
    result.experiment_name = "test-experiment"
    result.final_train_loss = 0.45
    result.final_eval_loss = 0.50
    result.best_eval_loss = 0.48
    result.total_steps = 1000
    result.training_time_seconds = 3600.0
    result.estimated_cost_usd = 5.50
    result.adapter_s3_uri = "s3://bucket/adapters/run-123"
    result.metrics = {"perplexity": 12.5}
    return result


class TestModelCardGenerator:
    """Tests for ModelCardGenerator."""

    def test_generate_returns_markdown(
        self,
        sample_config: TrainingJobConfig,
        sample_result: MagicMock,
    ) -> None:
        """Verify output is valid markdown with all 9 sections."""
        gen = ModelCardGenerator()
        content = gen.generate(sample_config, sample_result, eval_results={"mmlu": 0.75})

        expected_sections = [
            "Model Details",
            "Training Details",
            "Adapter Details",
            "Performance",
            "Intended Use",
            "Limitations",
            "Ethical Considerations",
            "Lineage",
            "How to Use",
        ]
        for section in expected_sections:
            assert f"## {section}" in content, f"Missing section: {section}"

    def test_generate_without_eval_results(
        self,
        sample_config: TrainingJobConfig,
        sample_result: MagicMock,
    ) -> None:
        """Verify Performance section says 'No evaluation results'."""
        gen = ModelCardGenerator()
        content = gen.generate(sample_config, sample_result, eval_results=None)

        assert "No evaluation results available." in content

    def test_generate_includes_lora_details(
        self,
        sample_config: TrainingJobConfig,
        sample_result: MagicMock,
    ) -> None:
        """Verify rank, alpha, target_modules in output."""
        gen = ModelCardGenerator()
        content = gen.generate(sample_config, sample_result)

        assert "64" in content  # rank
        assert "128" in content  # alpha
        assert "q_proj" in content
        assert "v_proj" in content

    def test_save_local(
        self,
        sample_config: TrainingJobConfig,
        sample_result: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Verify file written with correct content."""
        gen = ModelCardGenerator()
        content = gen.generate(sample_config, sample_result)
        output_path = str(tmp_dir / "README.md")
        gen.save(content, output_path)

        saved = Path(output_path).read_text()
        assert saved == content

    def test_save_json(
        self,
        sample_config: TrainingJobConfig,
        sample_result: MagicMock,
        tmp_dir: Path,
    ) -> None:
        """Verify JSON output matches expected schema."""
        gen = ModelCardGenerator()
        output_path = str(tmp_dir / "model_card.json")
        gen.save_json(sample_config, sample_result, {"mmlu": 0.75}, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "model_details" in data
        assert "training_results" in data
        assert "adapter_config" in data
        assert "eval_results" in data
        assert "lineage" in data
        assert data["adapter_config"]["r"] == 64

    @patch("src.monitoring.model_card.mlflow")
    @patch("src.monitoring.model_card._MLFLOW_AVAILABLE", True)
    def test_log_to_mlflow(self, mock_mlflow: MagicMock) -> None:
        """Mock mlflow, verify artifact logged."""
        gen = ModelCardGenerator()
        gen.log_to_mlflow("# Test Model Card")

        mock_mlflow.log_artifact.assert_called_once()

    @patch("subprocess.check_output", return_value=b"abc123def456\n")
    def test_lineage_includes_git_sha(
        self,
        mock_git: MagicMock,
        sample_config: TrainingJobConfig,
        sample_result: MagicMock,
    ) -> None:
        """Verify git SHA appears (mock subprocess)."""
        gen = ModelCardGenerator()
        content = gen.generate(sample_config, sample_result)

        assert "abc123def456" in content
