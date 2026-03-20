"""Unit tests for config.loader – YAML config loading with variable resolution."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.config.loader import (
    _resolve_ssm,
    _resolve_ssm_string,
    load_hpo_config,
    load_training_config,
)
from src.config.training import TrainingJobConfig


# ── helpers ─────────────────────────────────────────────────────


def _write_yaml(tmp_path: Path, data: dict, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return p


def _minimal_config(**overrides) -> dict:
    """Return a minimal valid YAML dict for TrainingJobConfig."""
    cfg: dict = {
        "experiment_name": "test-exp",
        "model": {"model_name_or_path": "meta-llama/Llama-2-7b-hf"},
        "sagemaker": {
            "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
        },
        "dataset_path": "./data/prepared/test",
        "output_s3_uri": "s3://bucket/output",
    }
    cfg.update(overrides)
    return cfg


# ── load_training_config ────────────────────────────────────────


class TestLoadTrainingConfig:
    def test_loads_minimal_yaml(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, _minimal_config())
        cfg = load_training_config(str(p))
        assert isinstance(cfg, TrainingJobConfig)
        assert cfg.experiment_name == "test-exp"

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_training_config("/nonexistent/path.yaml")

    def test_invalid_yaml_type(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_training_config(str(p))

    def test_validation_error(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["training"] = {"bf16": True, "fp16": True}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(Exception, match="bf16 and fp16"):
            load_training_config(str(p))

    def test_env_var_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_ROLE", "arn:aws:iam::111:role/R")
        data = _minimal_config()
        data["sagemaker"]["role_arn"] = "${MY_ROLE}"
        p = _write_yaml(tmp_path, data)
        cfg = load_training_config(str(p))
        assert cfg.sagemaker.role_arn == "arn:aws:iam::111:role/R"

    def test_env_var_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        data = _minimal_config()
        data["dataset_id"] = "${NONEXISTENT_VAR:-fallback-ds}"
        p = _write_yaml(tmp_path, data)
        cfg = load_training_config(str(p))
        assert cfg.dataset_id == "fallback-ds"

    def test_env_var_missing_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("REQUIRED_VAR", raising=False)
        data = _minimal_config()
        data["sagemaker"]["role_arn"] = "${REQUIRED_VAR}"
        p = _write_yaml(tmp_path, data)
        with pytest.raises(KeyError, match="REQUIRED_VAR"):
            load_training_config(str(p))

    def test_nested_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_BUCKET", "s3://my-bucket")
        monkeypatch.setenv("SM_ROLE", "arn:aws:iam::222:role/X")
        data = _minimal_config()
        data["output_s3_uri"] = "${MY_BUCKET}/output"
        data["sagemaker"]["role_arn"] = "${SM_ROLE}"
        p = _write_yaml(tmp_path, data)
        cfg = load_training_config(str(p))
        assert cfg.output_s3_uri == "s3://my-bucket/output"
        assert cfg.sagemaker.role_arn == "arn:aws:iam::222:role/X"

    def test_all_defaults_propagated(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, _minimal_config())
        cfg = load_training_config(str(p))
        assert cfg.model.torch_dtype == "bfloat16"
        assert cfg.quantization.bnb_4bit_quant_type == "nf4"
        assert cfg.lora.r == 64
        assert cfg.training.learning_rate == 2e-4
        assert cfg.training.bf16 is True

    def test_run_name_auto_generated(self, tmp_path: Path) -> None:
        p = _write_yaml(tmp_path, _minimal_config())
        cfg = load_training_config(str(p))
        assert cfg.run_name is not None
        assert cfg.run_name.startswith("test-exp-")

    def test_full_override(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["training"] = {
            "num_train_epochs": 5,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
        }
        data["lora"] = {"r": 32, "lora_alpha": 64}
        p = _write_yaml(tmp_path, data)
        cfg = load_training_config(str(p))
        assert cfg.training.num_train_epochs == 5
        assert cfg.training.learning_rate == 1e-4
        assert cfg.lora.r == 32


# ── SSM resolution ──────────────────────────────────────────────


class TestSSMResolution:
    def test_ssm_string_no_match(self) -> None:
        assert _resolve_ssm_string("plain-value") == "plain-value"

    def test_ssm_string_with_default(self) -> None:
        result = _resolve_ssm_string("${ssm:/missing/param:-fallback}")
        assert result == "fallback"

    @patch("src.config.loader._get_ssm_parameter")
    def test_ssm_string_resolved(self, mock_ssm: MagicMock) -> None:
        mock_ssm.return_value = "resolved-value"
        result = _resolve_ssm_string("${ssm:/my/param}")
        mock_ssm.assert_called_once_with("/my/param")
        assert result == "resolved-value"

    @patch("src.config.loader._get_ssm_parameter")
    def test_ssm_dict_recursive(self, mock_ssm: MagicMock) -> None:
        mock_ssm.return_value = "ssm-val"
        data = {
            "top": "${ssm:/a}",
            "nested": {"inner": "${ssm:/b}"},
            "list_field": ["${ssm:/c}", "plain"],
            "number": 42,
        }
        result = _resolve_ssm(data)
        assert result["top"] == "ssm-val"
        assert result["nested"]["inner"] == "ssm-val"
        assert result["list_field"] == ["ssm-val", "plain"]
        assert result["number"] == 42

    def test_ssm_missing_no_default_raises(self) -> None:
        with pytest.raises(Exception):
            _resolve_ssm_string("${ssm:/does/not/exist}")


# ── load_hpo_config ─────────────────────────────────────────────


class TestLoadHPOConfig:
    def test_loads_hpo_yaml(self, tmp_path: Path) -> None:
        data = {
            "base_config": "configs/training/qlora_llama3_8b.yaml",
            "objective": {"metric_name": "eval_loss", "type": "Minimize"},
            "strategy": "Bayesian",
            "max_jobs": 20,
            "max_parallel_jobs": 4,
            "hyperparameter_ranges": {
                "learning_rate": {
                    "type": "Continuous",
                    "min": 1e-5,
                    "max": 5e-4,
                    "scaling": "Logarithmic",
                },
            },
        }
        p = _write_yaml(tmp_path, data)
        result = load_hpo_config(str(p))
        assert result["strategy"] == "Bayesian"
        assert result["max_jobs"] == 20

    def test_hpo_env_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HPO_BASE", "configs/training/custom.yaml")
        data = {
            "base_config": "${HPO_BASE}",
            "strategy": "Bayesian",
            "max_jobs": 10,
        }
        p = _write_yaml(tmp_path, data)
        result = load_hpo_config(str(p))
        assert result["base_config"] == "configs/training/custom.yaml"


# ── concrete YAML files ────────────────────────────────────────


class TestConcreteYAMLFiles:
    """Validate that each shipped YAML config parses and validates correctly."""

    _CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "training"

    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "SAGEMAKER_ROLE_ARN",
            "arn:aws:iam::123456789012:role/SageMakerRole",
        )
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    @pytest.mark.parametrize(
        "filename",
        [
            "qlora_llama3_8b.yaml",
            "dora_llama3_8b.yaml",
            "qlora_llama3_70b.yaml",
            "qlora_mistral_7b.yaml",
        ],
    )
    def test_config_loads_and_validates(self, filename: str) -> None:
        cfg = load_training_config(str(self._CONFIGS_DIR / filename))
        assert isinstance(cfg, TrainingJobConfig)

    def test_qlora_llama3_8b_values(self) -> None:
        cfg = load_training_config(
            str(self._CONFIGS_DIR / "qlora_llama3_8b.yaml")
        )
        assert cfg.model.model_name_or_path == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert cfg.quantization.method == "qlora"
        assert cfg.lora.use_dora is False
        assert cfg.sagemaker.instance_type == "ml.g5.2xlarge"
        assert cfg.training.learning_rate == 2e-4

    def test_dora_llama3_8b_values(self) -> None:
        cfg = load_training_config(
            str(self._CONFIGS_DIR / "dora_llama3_8b.yaml")
        )
        assert cfg.quantization.method == "dora"
        assert cfg.lora.use_dora is True
        assert cfg.lora.r == 128
        assert cfg.training.learning_rate == 1e-4

    def test_qlora_llama3_70b_values(self) -> None:
        cfg = load_training_config(
            str(self._CONFIGS_DIR / "qlora_llama3_70b.yaml")
        )
        assert cfg.model.model_name_or_path == "meta-llama/Meta-Llama-3.1-70B-Instruct"
        assert cfg.sagemaker.instance_type == "ml.g5.48xlarge"
        assert cfg.training.per_device_train_batch_size == 1
        assert cfg.training.gradient_accumulation_steps == 16
        assert cfg.sagemaker.volume_size_gb == 500
        # FSDP configured via environment
        assert cfg.sagemaker.environment["ACCELERATE_USE_FSDP"] == "1"

    def test_qlora_mistral_7b_values(self) -> None:
        cfg = load_training_config(
            str(self._CONFIGS_DIR / "qlora_mistral_7b.yaml")
        )
        assert "mistral" in cfg.model.model_name_or_path.lower()
        assert cfg.sagemaker.instance_type == "ml.g5.2xlarge"

    def test_hpo_search_loads(self) -> None:
        result = load_hpo_config(
            str(self._CONFIGS_DIR / "hpo_search.yaml")
        )
        assert result["strategy"] == "Bayesian"
        assert result["max_jobs"] == 20
        assert result["max_parallel_jobs"] == 4
        assert "learning_rate" in result["hyperparameter_ranges"]
        assert result["objective"]["type"] == "Minimize"
        lr_range = result["hyperparameter_ranges"]["learning_rate"]
        assert lr_range["type"] == "Continuous"
        assert lr_range["scaling"] == "Logarithmic"
