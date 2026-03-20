"""Unit tests for training.sagemaker_launcher – SageMakerTrainingLauncher."""
from __future__ import annotations

import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# ── Create mock sagemaker modules before importing the launcher ──
# sagemaker SDK is not installed locally; we mock the package hierarchy
# so that the launcher's lazy ``from sagemaker.X import Y`` works.
_sagemaker_mod = types.ModuleType("sagemaker")
_sagemaker_hf_mod = types.ModuleType("sagemaker.huggingface")
_sagemaker_tuner_mod = types.ModuleType("sagemaker.tuner")

_sagemaker_hf_mod.HuggingFace = MagicMock  # type: ignore[attr-defined]
_sagemaker_tuner_mod.HyperparameterTuner = MagicMock  # type: ignore[attr-defined]
_sagemaker_tuner_mod.ContinuousParameter = MagicMock  # type: ignore[attr-defined]
_sagemaker_tuner_mod.IntegerParameter = MagicMock  # type: ignore[attr-defined]
_sagemaker_tuner_mod.CategoricalParameter = MagicMock  # type: ignore[attr-defined]

# Link submodules as attributes on the parent (required for mock._dot_lookup)
_sagemaker_mod.huggingface = _sagemaker_hf_mod  # type: ignore[attr-defined]
_sagemaker_mod.tuner = _sagemaker_tuner_mod  # type: ignore[attr-defined]

sys.modules.setdefault("sagemaker", _sagemaker_mod)
sys.modules.setdefault("sagemaker.huggingface", _sagemaker_hf_mod)
sys.modules.setdefault("sagemaker.tuner", _sagemaker_tuner_mod)

from src.config.training import (
    SageMakerConfig,
    TrainingJobConfig,
    VPCConfig,
)
from src.training.sagemaker_launcher import SageMakerTrainingLauncher
from src.training.trainer import TrainingResult


# ── helpers ─────────────────────────────────────────────────────


def _make_config(**overrides: Any) -> TrainingJobConfig:
    """Build a minimal valid TrainingJobConfig with optional overrides."""
    defaults: dict[str, Any] = {
        "experiment_name": "test-experiment",
        "model": {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"},
        "sagemaker": {
            "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
            "instance_type": "ml.g5.2xlarge",
        },
        "dataset_path": "/data/datasets/my-dataset",
        "output_s3_uri": "s3://my-bucket/output",
    }
    defaults.update(overrides)
    return TrainingJobConfig(**defaults)


def _make_launcher() -> SageMakerTrainingLauncher:
    """Create launcher with mocked AWS clients."""
    with patch("training.sagemaker_launcher.boto3") as mock_boto:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()
    return launcher


# ═══════════════════════════════════════════════════════════════
# _generate_job_name
# ═══════════════════════════════════════════════════════════════


class TestGenerateJobName:
    def test_contains_prefix(self) -> None:
        name = SageMakerTrainingLauncher._generate_job_name("my-exp")
        assert name.startswith("my-exp-")

    def test_max_length_63(self) -> None:
        name = SageMakerTrainingLauncher._generate_job_name("a" * 100)
        assert len(name) <= 63

    def test_alphanumeric_and_hyphens_only(self) -> None:
        name = SageMakerTrainingLauncher._generate_job_name("my_exp!@#$")
        assert all(c.isalnum() or c == "-" for c in name)

    def test_unique_names(self) -> None:
        names = {
            SageMakerTrainingLauncher._generate_job_name("exp") for _ in range(10)
        }
        assert len(names) == 10


# ═══════════════════════════════════════════════════════════════
# _upload_config
# ═══════════════════════════════════════════════════════════════


class TestUploadConfig:
    def test_uploads_yaml_to_s3(self) -> None:
        launcher = _make_launcher()
        config = _make_config()

        uri = launcher._upload_config(config)

        launcher._s3_client.put_object.assert_called_once()
        call_kwargs = launcher._s3_client.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "my-bucket"
        assert call_kwargs["ContentType"] == "application/x-yaml"
        assert uri.startswith("s3://my-bucket/")
        assert uri.endswith("config.yaml")

    def test_config_uri_contains_run_name(self) -> None:
        launcher = _make_launcher()
        config = _make_config()

        uri = launcher._upload_config(config)

        # run_name is auto-generated and should appear in the URI
        assert config.run_name in uri or config.experiment_name in uri


# ═══════════════════════════════════════════════════════════════
# _extract_final_metrics
# ═══════════════════════════════════════════════════════════════


class TestExtractFinalMetrics:
    def test_extracts_metrics(self) -> None:
        resp = {
            "FinalMetricDataList": [
                {"MetricName": "train_loss", "Value": 0.45, "Timestamp": 0},
                {"MetricName": "eval_loss", "Value": 0.52, "Timestamp": 0},
            ],
        }
        metrics = SageMakerTrainingLauncher._extract_final_metrics(resp)
        assert metrics["train_loss"] == pytest.approx(0.45)
        assert metrics["eval_loss"] == pytest.approx(0.52)

    def test_empty_when_no_metrics(self) -> None:
        metrics = SageMakerTrainingLauncher._extract_final_metrics({})
        assert metrics == {}


# ═══════════════════════════════════════════════════════════════
# _parse_hp_ranges
# ═══════════════════════════════════════════════════════════════


class TestParseHpRanges:
    def test_continuous(self) -> None:
        mock_cls = MagicMock()
        ranges = {
            "learning_rate": {
                "type": "Continuous",
                "min": 1e-5,
                "max": 5e-4,
                "scaling": "Logarithmic",
            },
        }
        result = SageMakerTrainingLauncher._parse_hp_ranges(
            ranges, mock_cls, MagicMock, MagicMock,
        )
        mock_cls.assert_called_once_with(
            min_value=1e-5, max_value=5e-4, scaling_type="Logarithmic",
        )
        assert "learning_rate" in result

    def test_integer(self) -> None:
        mock_cls = MagicMock()
        ranges = {
            "epochs": {"type": "Integer", "min": 1, "max": 10},
        }
        result = SageMakerTrainingLauncher._parse_hp_ranges(
            ranges, MagicMock, mock_cls, MagicMock,
        )
        mock_cls.assert_called_once_with(min_value=1, max_value=10)
        assert "epochs" in result

    def test_categorical(self) -> None:
        mock_cls = MagicMock()
        ranges = {
            "batch_size": {"type": "Categorical", "values": [2, 4, 8]},
        }
        result = SageMakerTrainingLauncher._parse_hp_ranges(
            ranges, MagicMock, MagicMock, mock_cls,
        )
        mock_cls.assert_called_once_with([2, 4, 8])
        assert "batch_size" in result

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown HP range type"):
            SageMakerTrainingLauncher._parse_hp_ranges(
                {"x": {"type": "Unknown"}}, MagicMock, MagicMock, MagicMock,
            )

    def test_mixed_ranges(self) -> None:
        cont_cls = MagicMock()
        int_cls = MagicMock()
        cat_cls = MagicMock()
        ranges = {
            "lr": {"type": "Continuous", "min": 0.0, "max": 1.0},
            "epochs": {"type": "Integer", "min": 1, "max": 10},
            "bs": {"type": "Categorical", "values": [2, 4]},
        }
        result = SageMakerTrainingLauncher._parse_hp_ranges(
            ranges, cont_cls, int_cls, cat_cls,
        )
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════
# launch
# ═══════════════════════════════════════════════════════════════


class TestLaunch:
    @patch("training.sagemaker_launcher.boto3")
    def test_launch_calls_estimator_fit(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()

        mock_estimator = MagicMock()
        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://my-bucket/config.yaml",
        ):
            with patch(
                "sagemaker.huggingface.HuggingFace",
                return_value=mock_estimator,
            ) as mock_hf_cls:
                config = _make_config()
                job_name = launcher.launch(config)

        # Estimator was constructed
        mock_hf_cls.assert_called_once()
        hf_kwargs = mock_hf_cls.call_args.kwargs

        assert hf_kwargs["entry_point"] == "train_entry.py"
        assert hf_kwargs["source_dir"] == "src/"
        assert hf_kwargs["instance_type"] == "ml.g5.2xlarge"
        assert hf_kwargs["instance_count"] == 1
        assert hf_kwargs["role"] == "arn:aws:iam::123456789012:role/SageMakerRole"
        assert hf_kwargs["transformers_version"] == "4.44.0"
        assert hf_kwargs["pytorch_version"] == "2.2.0"
        assert hf_kwargs["py_version"] == "py310"
        assert hf_kwargs["volume_size"] == 200
        assert hf_kwargs["use_spot_instances"] is False
        assert hf_kwargs["max_wait"] is None  # no spot
        assert hf_kwargs["distribution"] is None  # single instance

        # fit was called
        mock_estimator.fit.assert_called_once()
        fit_kwargs = mock_estimator.fit.call_args.kwargs
        assert "train" in fit_kwargs["inputs"]
        assert "validation" in fit_kwargs["inputs"]
        assert fit_kwargs["wait"] is False
        assert isinstance(job_name, str)

    @patch("training.sagemaker_launcher.boto3")
    def test_launch_with_spot_instances(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()

        config = _make_config(
            sagemaker={
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                "instance_type": "ml.g5.2xlarge",
                "use_spot_instances": True,
                "max_wait_seconds": 172800,
                "checkpoint_s3_uri": "s3://my-bucket/checkpoints",
            },
        )

        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/config.yaml",
        ):
            with patch(
                "sagemaker.huggingface.HuggingFace",
                return_value=MagicMock(),
            ) as mock_hf_cls:
                launcher.launch(config)

        hf_kwargs = mock_hf_cls.call_args.kwargs
        assert hf_kwargs["use_spot_instances"] is True
        assert hf_kwargs["max_wait"] == 172800
        assert hf_kwargs["checkpoint_s3_uri"] == "s3://my-bucket/checkpoints"

    @patch("training.sagemaker_launcher.boto3")
    def test_launch_with_vpc(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()

        config = _make_config(
            sagemaker={
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                "vpc_config": {
                    "subnets": ["subnet-abc"],
                    "security_group_ids": ["sg-123"],
                },
            },
        )

        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/config.yaml",
        ):
            with patch(
                "sagemaker.huggingface.HuggingFace",
                return_value=MagicMock(),
            ) as mock_hf_cls:
                launcher.launch(config)

        hf_kwargs = mock_hf_cls.call_args.kwargs
        assert hf_kwargs["subnets"] == ["subnet-abc"]
        assert hf_kwargs["security_group_ids"] == ["sg-123"]

    @patch("training.sagemaker_launcher.boto3")
    def test_launch_multinode_distribution(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()

        config = _make_config(
            sagemaker={
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                "instance_count": 4,
            },
        )

        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/config.yaml",
        ):
            with patch(
                "sagemaker.huggingface.HuggingFace",
                return_value=MagicMock(),
            ) as mock_hf_cls:
                launcher.launch(config)

        hf_kwargs = mock_hf_cls.call_args.kwargs
        assert hf_kwargs["instance_count"] == 4
        assert hf_kwargs["distribution"] == {
            "torch_distributed": {"enabled": True},
        }

    @patch("training.sagemaker_launcher.boto3")
    def test_launch_environment_includes_config_uri(
        self, mock_boto: MagicMock,
    ) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()
        config = _make_config(
            sagemaker={
                "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                "environment": {"CUSTOM_VAR": "hello"},
            },
        )

        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/output/configs/test/config.yaml",
        ):
            with patch(
                "sagemaker.huggingface.HuggingFace",
                return_value=MagicMock(),
            ) as mock_hf_cls:
                launcher.launch(config)

        env = mock_hf_cls.call_args.kwargs["environment"]
        assert env["CONFIG_S3_URI"] == "s3://bucket/output/configs/test/config.yaml"
        assert env["DATASET_ID"] == "/data/datasets/my-dataset"
        assert env["HF_TOKEN"] == "{{resolve:secretsmanager:hf-token}}"
        assert env["CUSTOM_VAR"] == "hello"

    @patch("training.sagemaker_launcher.boto3")
    def test_launch_tags(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()
        config = _make_config()

        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/config.yaml",
        ):
            with patch(
                "sagemaker.huggingface.HuggingFace",
                return_value=MagicMock(),
            ) as mock_hf_cls:
                launcher.launch(config)

        tags = mock_hf_cls.call_args.kwargs["tags"]
        tag_keys = [t["Key"] for t in tags]
        assert "Project" in tag_keys
        assert "RunName" in tag_keys
        project_tag = next(t for t in tags if t["Key"] == "Project")
        assert project_tag["Value"] == "test-experiment"

    @patch("training.sagemaker_launcher.boto3")
    def test_launch_returns_job_name(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()
        config = _make_config()

        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/config.yaml",
        ):
            with patch(
                "sagemaker.huggingface.HuggingFace",
                return_value=MagicMock(),
            ):
                job_name = launcher.launch(config)

        assert isinstance(job_name, str)
        assert len(job_name) > 0
        assert len(job_name) <= 63


# ═══════════════════════════════════════════════════════════════
# wait_for_job
# ═══════════════════════════════════════════════════════════════


class TestWaitForJob:
    @patch("training.sagemaker_launcher.boto3")
    @patch("training.sagemaker_launcher.time.sleep")
    def test_completed_job_returns_result(
        self, mock_sleep: MagicMock, mock_boto: MagicMock,
    ) -> None:
        mock_sm = MagicMock()
        mock_boto.client.return_value = mock_sm
        launcher = SageMakerTrainingLauncher()

        mock_sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "SecondaryStatus": "Completed",
            "BillableTimeInSeconds": 3600,
            "ModelArtifacts": {
                "S3ModelArtifacts": "s3://bucket/output/model.tar.gz",
            },
            "FinalMetricDataList": [
                {"MetricName": "train_loss", "Value": 0.3, "Timestamp": 0},
                {"MetricName": "eval_loss", "Value": 0.4, "Timestamp": 0},
            ],
        }

        result = launcher.wait_for_job("my-job-123")

        assert isinstance(result, TrainingResult)
        assert result.run_id == "my-job-123"
        assert result.final_train_loss == pytest.approx(0.3)
        assert result.final_eval_loss == pytest.approx(0.4)
        assert result.training_time_seconds == 3600.0
        assert result.adapter_s3_uri == "s3://bucket/output/model.tar.gz"
        mock_sleep.assert_not_called()

    @patch("training.sagemaker_launcher.boto3")
    @patch("training.sagemaker_launcher.time.sleep")
    def test_polls_until_completed(
        self, mock_sleep: MagicMock, mock_boto: MagicMock,
    ) -> None:
        mock_sm = MagicMock()
        mock_boto.client.return_value = mock_sm
        launcher = SageMakerTrainingLauncher()

        mock_sm.describe_training_job.side_effect = [
            {"TrainingJobStatus": "InProgress", "SecondaryStatus": "Training"},
            {"TrainingJobStatus": "InProgress", "SecondaryStatus": "Training"},
            {
                "TrainingJobStatus": "Completed",
                "BillableTimeInSeconds": 1800,
                "ModelArtifacts": {"S3ModelArtifacts": "s3://b/m"},
                "FinalMetricDataList": [],
            },
        ]

        result = launcher.wait_for_job("poll-job", poll_interval=10)

        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10)
        assert isinstance(result, TrainingResult)

    @patch("training.sagemaker_launcher.boto3")
    @patch("training.sagemaker_launcher.time.sleep")
    def test_failed_job_raises(
        self, mock_sleep: MagicMock, mock_boto: MagicMock,
    ) -> None:
        mock_sm = MagicMock()
        mock_logs = MagicMock()

        def client_factory(service: str) -> MagicMock:
            if service == "logs":
                return mock_logs
            return mock_sm

        mock_boto.client.side_effect = client_factory
        launcher = SageMakerTrainingLauncher()
        # Reset _sm_client to the sagemaker mock
        launcher._sm_client = mock_sm

        mock_sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Failed",
            "FailureReason": "OOM on GPU 0",
        }
        mock_logs.describe_log_streams.return_value = {"logStreams": []}

        with pytest.raises(RuntimeError, match="OOM on GPU 0"):
            launcher.wait_for_job("fail-job")

    @patch("training.sagemaker_launcher.boto3")
    @patch("training.sagemaker_launcher.time.sleep")
    def test_stopped_job_raises(
        self, mock_sleep: MagicMock, mock_boto: MagicMock,
    ) -> None:
        mock_sm = MagicMock()
        mock_boto.client.return_value = mock_sm
        launcher = SageMakerTrainingLauncher()

        mock_sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Stopped",
        }

        with pytest.raises(RuntimeError, match="was stopped"):
            launcher.wait_for_job("stop-job")

    @patch("training.sagemaker_launcher.boto3")
    @patch("training.sagemaker_launcher.time.sleep")
    def test_completed_with_no_metrics(
        self, mock_sleep: MagicMock, mock_boto: MagicMock,
    ) -> None:
        mock_sm = MagicMock()
        mock_boto.client.return_value = mock_sm
        launcher = SageMakerTrainingLauncher()

        mock_sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "BillableTimeInSeconds": 100,
            "ModelArtifacts": {"S3ModelArtifacts": "s3://b/m"},
        }

        result = launcher.wait_for_job("no-metrics-job")

        assert result.final_train_loss == 0.0
        assert result.final_eval_loss == 0.0
        assert result.total_steps == 0
        assert result.metrics == {}


# ═══════════════════════════════════════════════════════════════
# _fetch_cloudwatch_logs
# ═══════════════════════════════════════════════════════════════


class TestFetchCloudwatchLogs:
    @patch("training.sagemaker_launcher.boto3")
    def test_returns_log_lines(self, mock_boto: MagicMock) -> None:
        mock_sm = MagicMock()
        mock_logs = MagicMock()

        call_count = {"n": 0}

        def client_factory(service: str) -> MagicMock:
            call_count["n"] += 1
            if service == "logs":
                return mock_logs
            return mock_sm

        mock_boto.client.side_effect = client_factory
        launcher = SageMakerTrainingLauncher()

        mock_logs.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "my-job/algo-1"}],
        }
        mock_logs.get_log_events.return_value = {
            "events": [
                {"message": "Epoch 1/3"},
                {"message": "Loss: 0.45"},
            ],
        }

        result = launcher._fetch_cloudwatch_logs("my-job")

        assert "Epoch 1/3" in result
        assert "Loss: 0.45" in result

    @patch("training.sagemaker_launcher.boto3")
    def test_no_streams_returns_placeholder(self, mock_boto: MagicMock) -> None:
        mock_sm = MagicMock()
        mock_logs = MagicMock()

        def client_factory(service: str) -> MagicMock:
            if service == "logs":
                return mock_logs
            return mock_sm

        mock_boto.client.side_effect = client_factory
        launcher = SageMakerTrainingLauncher()

        mock_logs.describe_log_streams.return_value = {"logStreams": []}

        result = launcher._fetch_cloudwatch_logs("my-job")
        assert "no log streams found" in result

    @patch("training.sagemaker_launcher.boto3")
    def test_exception_returns_error_message(self, mock_boto: MagicMock) -> None:
        mock_sm = MagicMock()
        mock_logs = MagicMock()

        def client_factory(service: str) -> MagicMock:
            if service == "logs":
                return mock_logs
            return mock_sm

        mock_boto.client.side_effect = client_factory
        launcher = SageMakerTrainingLauncher()

        mock_logs.describe_log_streams.side_effect = Exception("Access denied")

        result = launcher._fetch_cloudwatch_logs("my-job")
        assert "failed to fetch logs" in result


# ═══════════════════════════════════════════════════════════════
# launch_hpo
# ═══════════════════════════════════════════════════════════════


class TestLaunchHpo:
    @patch("training.sagemaker_launcher.boto3")
    def test_launch_hpo_creates_tuner(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()
        config = _make_config()

        hpo_config = {
            "objective": {"metric_name": "eval_loss", "type": "Minimize"},
            "strategy": "Bayesian",
            "max_jobs": 10,
            "max_parallel_jobs": 2,
            "hyperparameter_ranges": {
                "learning_rate": {
                    "type": "Continuous",
                    "min": 1e-5,
                    "max": 5e-4,
                    "scaling": "Logarithmic",
                },
                "lora_r": {
                    "type": "Categorical",
                    "values": [16, 32, 64],
                },
            },
        }

        mock_tuner = MagicMock()
        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/config.yaml",
        ):
            with patch("sagemaker.huggingface.HuggingFace", return_value=MagicMock()):
                with patch(
                    "sagemaker.tuner.HyperparameterTuner",
                    return_value=mock_tuner,
                ) as mock_tuner_cls:
                    tuner_name = launcher.launch_hpo(config, hpo_config)

        # Tuner was created
        mock_tuner_cls.assert_called_once()
        tuner_kwargs = mock_tuner_cls.call_args.kwargs
        assert tuner_kwargs["objective_metric_name"] == "eval_loss"
        assert tuner_kwargs["objective_type"] == "Minimize"
        assert tuner_kwargs["strategy"] == "Bayesian"
        assert tuner_kwargs["max_jobs"] == 10
        assert tuner_kwargs["max_parallel_jobs"] == 2

        # fit was called
        mock_tuner.fit.assert_called_once()
        fit_kwargs = mock_tuner.fit.call_args.kwargs
        assert "train" in fit_kwargs["inputs"]
        assert fit_kwargs["wait"] is False

        assert isinstance(tuner_name, str)
        assert len(tuner_name) <= 63

    @patch("training.sagemaker_launcher.boto3")
    def test_launch_hpo_default_values(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()
        config = _make_config()

        # Minimal HPO config — use defaults
        hpo_config: dict[str, Any] = {
            "hyperparameter_ranges": {},
        }

        mock_tuner = MagicMock()
        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/config.yaml",
        ):
            with patch("sagemaker.huggingface.HuggingFace", return_value=MagicMock()):
                with patch(
                    "sagemaker.tuner.HyperparameterTuner",
                    return_value=mock_tuner,
                ) as mock_tuner_cls:
                    launcher.launch_hpo(config, hpo_config)

        tuner_kwargs = mock_tuner_cls.call_args.kwargs
        assert tuner_kwargs["objective_metric_name"] == "eval_loss"
        assert tuner_kwargs["objective_type"] == "Minimize"
        assert tuner_kwargs["strategy"] == "Bayesian"
        assert tuner_kwargs["max_jobs"] == 20
        assert tuner_kwargs["max_parallel_jobs"] == 4

    @patch("training.sagemaker_launcher.boto3")
    def test_launch_hpo_with_integer_ranges(self, mock_boto: MagicMock) -> None:
        mock_boto.client.return_value = MagicMock()
        launcher = SageMakerTrainingLauncher()
        config = _make_config()

        hpo_config = {
            "hyperparameter_ranges": {
                "num_epochs": {"type": "Integer", "min": 1, "max": 5},
            },
        }

        mock_tuner = MagicMock()
        with patch(
            "training.sagemaker_launcher.SageMakerTrainingLauncher._upload_config",
            return_value="s3://bucket/config.yaml",
        ):
            with patch("sagemaker.huggingface.HuggingFace", return_value=MagicMock()):
                with patch(
                    "sagemaker.tuner.HyperparameterTuner",
                    return_value=mock_tuner,
                ) as mock_tuner_cls:
                    launcher.launch_hpo(config, hpo_config)

        hp_ranges = mock_tuner_cls.call_args.kwargs["hyperparameter_ranges"]
        assert "num_epochs" in hp_ranges
