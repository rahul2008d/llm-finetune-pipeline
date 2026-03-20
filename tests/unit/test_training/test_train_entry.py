"""Unit tests for training.train_entry – SageMaker container entry point."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.training.train_entry import (
    _download_config,
    _override_dataset_paths,
    _parse_sm_env,
    main,
)


# ═══════════════════════════════════════════════════════════════
# _parse_sm_env
# ═══════════════════════════════════════════════════════════════


class TestParseSMEnv:
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            env = _parse_sm_env()

        assert env["model_dir"] == "/opt/ml/model"
        assert env["train_dir"] == "/opt/ml/input/data/train"
        assert env["validation_dir"] == "/opt/ml/input/data/validation"
        assert env["output_dir"] == "/opt/ml/output/data"
        assert env["config_s3_uri"] == ""
        assert env["num_gpus"] == "1"

    def test_custom_values(self) -> None:
        custom = {
            "SM_MODEL_DIR": "/my/model",
            "SM_CHANNEL_TRAIN": "/my/train",
            "SM_CHANNEL_VALIDATION": "/my/val",
            "SM_OUTPUT_DATA_DIR": "/my/output",
            "CONFIG_S3_URI": "s3://bucket/config.yaml",
            "SM_NUM_GPUS": "4",
        }
        with patch.dict(os.environ, custom, clear=True):
            env = _parse_sm_env()

        assert env["model_dir"] == "/my/model"
        assert env["train_dir"] == "/my/train"
        assert env["validation_dir"] == "/my/val"
        assert env["output_dir"] == "/my/output"
        assert env["config_s3_uri"] == "s3://bucket/config.yaml"
        assert env["num_gpus"] == "4"


# ═══════════════════════════════════════════════════════════════
# _download_config
# ═══════════════════════════════════════════════════════════════


class TestDownloadConfig:
    def test_raises_on_empty_uri(self) -> None:
        with pytest.raises(ValueError, match="CONFIG_S3_URI"):
            _download_config("")

    @patch("training.train_entry.boto3")
    def test_downloads_yaml(self, mock_boto: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_boto.client.return_value = mock_s3

        config_dict = {"experiment_name": "test", "dataset_id": "ds-1"}
        yaml_body = yaml.dump(config_dict)
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=yaml_body.encode())),
        }

        result = _download_config("s3://my-bucket/configs/config.yaml")

        mock_s3.get_object.assert_called_once_with(
            Bucket="my-bucket", Key="configs/config.yaml",
        )
        assert result["experiment_name"] == "test"
        assert result["dataset_id"] == "ds-1"

    @patch("training.train_entry.boto3")
    def test_parses_bucket_and_key(self, mock_boto: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_boto.client.return_value = mock_s3
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b"key: val")),
        }

        _download_config("s3://deep-bucket/path/to/config.yaml")

        mock_s3.get_object.assert_called_once_with(
            Bucket="deep-bucket", Key="path/to/config.yaml",
        )


# ═══════════════════════════════════════════════════════════════
# _override_dataset_paths
# ═══════════════════════════════════════════════════════════════


class TestOverrideDatasetPaths:
    def test_overrides_dataset_path(self) -> None:
        config = {"dataset_id": "original", "other": "value"}
        result = _override_dataset_paths(
            config, "/opt/ml/input/data/train", "/opt/ml/input/data/validation",
        )
        assert result["dataset_path"] == "/opt/ml/input/data/train"
        assert "dataset_id" not in result
        assert result["_validation_dir"] == "/opt/ml/input/data/validation"
        assert result["other"] == "value"

    def test_preserves_other_keys(self) -> None:
        config = {
            "dataset_path": "old",
            "experiment_name": "exp",
            "model": {"key": "val"},
        }
        result = _override_dataset_paths(config, "/train", "/val")
        assert result["experiment_name"] == "exp"
        assert result["model"] == {"key": "val"}


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════


class TestMain:
    def _build_config_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": "sm-test",
            "model": {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"},
            "sagemaker": {
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
            },
            "dataset_path": "./data/prepared/test",
            "output_s3_uri": "s3://bucket/output",
        }

    @patch("training.train_entry.shutil.copy2")
    @patch("training.train_entry.FineTuneTrainer")
    @patch("training.train_entry.TrainingJobConfig")
    @patch("training.train_entry._download_config")
    def test_main_happy_path(
        self,
        mock_download: MagicMock,
        mock_config_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_copy: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()

            # Mock config download
            mock_download.return_value = self._build_config_dict()

            # Mock TrainingJobConfig.model_validate
            mock_config = MagicMock()
            mock_config_cls.model_validate.return_value = mock_config

            # Mock trainer result — adapter_s3_uri is an S3 path (not local)
            mock_result = MagicMock()
            mock_result.adapter_s3_uri = "s3://bucket/adapter"
            mock_result.run_id = "run-123"
            mock_result.model_dump.return_value = {
                "run_id": "run-123",
                "experiment_name": "sm-test",
                "final_train_loss": 0.3,
            }
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = mock_result
            mock_trainer_cls.return_value = mock_trainer

            sm_env = {
                "SM_MODEL_DIR": str(model_dir),
                "SM_CHANNEL_TRAIN": "/opt/ml/input/data/train",
                "SM_CHANNEL_VALIDATION": "/opt/ml/input/data/validation",
                "SM_OUTPUT_DATA_DIR": "/opt/ml/output",
                "CONFIG_S3_URI": "s3://bucket/config.yaml",
                "SM_NUM_GPUS": "1",
            }

            with patch.dict(os.environ, sm_env, clear=True):
                main()

            # Config was downloaded
            mock_download.assert_called_once_with("s3://bucket/config.yaml")

            # Trainer was created and trained
            mock_trainer_cls.assert_called_once_with(mock_config)
            mock_trainer.train.assert_called_once()

            # Result JSON was written
            result_path = model_dir / "training_result.json"
            assert result_path.exists()
            written = json.loads(result_path.read_text())
            assert written["run_id"] == "run-123"

    @patch("training.train_entry.shutil.copy2")
    @patch("training.train_entry.FineTuneTrainer")
    @patch("training.train_entry.TrainingJobConfig")
    @patch("training.train_entry._download_config")
    def test_main_copies_local_adapter(
        self,
        mock_download: MagicMock,
        mock_config_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_copy: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()

            # Create a local adapter dir with files
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_model.bin").write_text("weights")
            (adapter_dir / "config.json").write_text("{}")

            mock_download.return_value = self._build_config_dict()
            mock_config_cls.model_validate.return_value = MagicMock()

            mock_result = MagicMock()
            mock_result.adapter_s3_uri = str(adapter_dir)
            mock_result.run_id = "run-local"
            mock_result.model_dump.return_value = {"run_id": "run-local"}

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = mock_result
            mock_trainer_cls.return_value = mock_trainer

            sm_env = {
                "SM_MODEL_DIR": str(model_dir),
                "SM_CHANNEL_TRAIN": "/data/train",
                "SM_CHANNEL_VALIDATION": "/data/val",
                "CONFIG_S3_URI": "s3://bucket/config.yaml",
            }

            with patch.dict(os.environ, sm_env, clear=True):
                main()

            # copy2 was called for the adapter files
            assert mock_copy.call_count == 2

    @patch("training.train_entry.FineTuneTrainer")
    @patch("training.train_entry.TrainingJobConfig")
    @patch("training.train_entry._download_config")
    def test_main_overrides_dataset_path(
        self,
        mock_download: MagicMock,
        mock_config_cls: MagicMock,
        mock_trainer_cls: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()

            config_dict = self._build_config_dict()
            mock_download.return_value = config_dict

            mock_config = MagicMock()
            mock_config_cls.model_validate.return_value = mock_config

            mock_result = MagicMock()
            mock_result.adapter_s3_uri = "s3://bucket/adapter"
            mock_result.run_id = "run-1"
            mock_result.model_dump.return_value = {"run_id": "run-1"}
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = mock_result
            mock_trainer_cls.return_value = mock_trainer

            sm_env = {
                "SM_MODEL_DIR": str(model_dir),
                "SM_CHANNEL_TRAIN": "/custom/train",
                "SM_CHANNEL_VALIDATION": "/custom/val",
                "CONFIG_S3_URI": "s3://bucket/config.yaml",
            }

            with patch.dict(os.environ, sm_env, clear=True):
                main()

            # Check that model_validate was called with overridden dataset_path
            validate_arg = mock_config_cls.model_validate.call_args[0][0]
            assert validate_arg["dataset_path"] == "/custom/train"
            assert "dataset_id" not in validate_arg
            assert validate_arg["_validation_dir"] == "/custom/val"

    @patch("training.train_entry._download_config")
    def test_main_raises_on_missing_config_uri(
        self, mock_download: MagicMock,
    ) -> None:
        sm_env = {
            "SM_MODEL_DIR": "/opt/ml/model",
            "CONFIG_S3_URI": "",
        }
        mock_download.side_effect = ValueError("CONFIG_S3_URI")

        with patch.dict(os.environ, sm_env, clear=True):
            with pytest.raises(ValueError, match="CONFIG_S3_URI"):
                main()
