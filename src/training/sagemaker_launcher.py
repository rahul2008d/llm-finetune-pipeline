"""Launch and monitor SageMaker training jobs and HPO tuners."""
from __future__ import annotations

import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import structlog
import yaml

from src.config.training import TrainingJobConfig
from src.training.trainer import TrainingResult

logger = structlog.get_logger(__name__)


class SageMakerTrainingLauncher:
    """Orchestrates SageMaker training jobs and HPO tuners."""

    def __init__(self) -> None:
        self._sm_client = boto3.client("sagemaker")
        self._s3_client = boto3.client("s3")

    # ── public API ──────────────────────────────────────────────

    def launch(self, config: TrainingJobConfig) -> str:
        """Build a HuggingFace Estimator, upload config, and start a training job.

        Returns:
            The SageMaker job name for tracking.
        """
        from sagemaker.huggingface import HuggingFace

        job_name = self._generate_job_name(config.experiment_name)

        # Upload full config YAML to S3
        uploaded_config_uri = self._upload_config(config)

        # Build environment
        dataset_ref = config.dataset_id or config.dataset_path or ""
        environment = {
            **config.sagemaker.environment,
            "CONFIG_S3_URI": uploaded_config_uri,
            "DATASET_ID": dataset_ref,
            "HF_TOKEN": "{{resolve:secretsmanager:hf-token}}",
        }

        # Tags
        tags = [
            {"Key": "Project", "Value": config.experiment_name},
            {"Key": "RunName", "Value": config.run_name or job_name},
        ]

        # Distribution for multi-node
        distribution = (
            {"torch_distributed": {"enabled": True}}
            if config.sagemaker.instance_count > 1
            else None
        )

        # VPC kwargs
        subnets = None
        security_group_ids = None
        if config.sagemaker.vpc_config is not None:
            subnets = config.sagemaker.vpc_config.subnets
            security_group_ids = config.sagemaker.vpc_config.security_group_ids

        # Spot max_wait
        max_wait = (
            config.sagemaker.max_wait_seconds
            if config.sagemaker.use_spot_instances
            else None
        )

        estimator = HuggingFace(
            entry_point="train_entry.py",
            source_dir="src/training/",
            instance_type=config.sagemaker.instance_type,
            instance_count=config.sagemaker.instance_count,
            role=config.sagemaker.role_arn,
            transformers_version="4.36.0",
            pytorch_version="2.1.0",
            py_version="py310",
            volume_size=config.sagemaker.volume_size_gb,
            max_run=config.sagemaker.max_run_seconds,
            use_spot_instances=config.sagemaker.use_spot_instances,
            max_wait=max_wait,
            checkpoint_s3_uri=config.sagemaker.checkpoint_s3_uri,
            subnets=subnets,
            security_group_ids=security_group_ids,
            environment=environment,
            tags=tags,
            distribution=distribution,
        )

        # Build input channels
        s3_base = config.output_s3_uri.rstrip("/")
        channels = {
            "train": f"{s3_base}/datasets/{dataset_ref}/train/",
            "validation": f"{s3_base}/datasets/{dataset_ref}/validation/",
        }

        logger.info(
            "Launching SageMaker training job",
            job_name=job_name,
            instance_type=config.sagemaker.instance_type,
            instance_count=config.sagemaker.instance_count,
        )

        estimator.fit(inputs=channels, job_name=job_name, wait=False)
        return job_name

    def wait_for_job(
        self,
        job_name: str,
        poll_interval: int = 60,
    ) -> TrainingResult:
        """Poll SageMaker until the training job reaches a terminal state.

        Logs progress from CloudWatch when available.  On failure, fetches
        CloudWatch logs and raises with context.
        """
        terminal_states = {"Completed", "Failed", "Stopped"}

        while True:
            resp = self._sm_client.describe_training_job(
                TrainingJobName=job_name,
            )
            status = resp["TrainingJobStatus"]
            secondary = resp.get("SecondaryStatus", "")

            logger.info(
                "Job status",
                job_name=job_name,
                status=status,
                secondary=secondary,
            )

            if status in terminal_states:
                break

            time.sleep(poll_interval)

        if status == "Failed":
            failure_reason = resp.get("FailureReason", "Unknown")
            log_tail = self._fetch_cloudwatch_logs(job_name, tail=50)
            raise RuntimeError(
                f"Training job {job_name} failed: {failure_reason}\n"
                f"--- CloudWatch tail ---\n{log_tail}"
            )

        if status == "Stopped":
            raise RuntimeError(f"Training job {job_name} was stopped.")

        # Build TrainingResult from job metadata
        metrics = self._extract_final_metrics(resp)
        billable = resp.get("BillableTimeInSeconds", 0)
        model_artifacts = resp.get("ModelArtifacts", {}).get(
            "S3ModelArtifacts", ""
        )

        return TrainingResult(
            run_id=job_name,
            experiment_name=resp.get(
                "ExperimentConfig", {},
            ).get("ExperimentName", job_name),
            final_train_loss=metrics.get("train_loss", 0.0),
            final_eval_loss=metrics.get("eval_loss", 0.0),
            best_eval_loss=metrics.get("eval_loss", 0.0),
            total_steps=int(metrics.get("total_steps", 0)),
            training_time_seconds=float(billable),
            estimated_cost_usd=0.0,
            adapter_s3_uri=model_artifacts,
            metrics=metrics,
        )

    def launch_hpo(
        self,
        config: TrainingJobConfig,
        hpo_config: dict[str, Any],
    ) -> str:
        """Create a HyperparameterTuner from *hpo_config* and launch it.

        Args:
            config: Base training job configuration.
            hpo_config: Raw HPO search-space dict (from ``load_hpo_config``).

        Returns:
            The tuner name for tracking.
        """
        from sagemaker.huggingface import HuggingFace
        from sagemaker.tuner import (
            CategoricalParameter,
            ContinuousParameter,
            HyperparameterTuner,
            IntegerParameter,
        )

        # Build base estimator (same as launch, but we won't call .fit)
        uploaded_config_uri = self._upload_config(config)
        dataset_ref = config.dataset_id or config.dataset_path or ""
        environment = {
            **config.sagemaker.environment,
            "CONFIG_S3_URI": uploaded_config_uri,
            "DATASET_ID": dataset_ref,
            "HF_TOKEN": "{{resolve:secretsmanager:hf-token}}",
        }

        subnets = None
        security_group_ids = None
        if config.sagemaker.vpc_config is not None:
            subnets = config.sagemaker.vpc_config.subnets
            security_group_ids = config.sagemaker.vpc_config.security_group_ids

        max_wait = (
            config.sagemaker.max_wait_seconds
            if config.sagemaker.use_spot_instances
            else None
        )

        estimator = HuggingFace(
            entry_point="train_entry.py",
            source_dir="src/training/",
            instance_type=config.sagemaker.instance_type,
            instance_count=config.sagemaker.instance_count,
            role=config.sagemaker.role_arn,
            transformers_version="4.36.0",
            pytorch_version="2.1.0",
            py_version="py310",
            volume_size=config.sagemaker.volume_size_gb,
            max_run=config.sagemaker.max_run_seconds,
            use_spot_instances=config.sagemaker.use_spot_instances,
            max_wait=max_wait,
            checkpoint_s3_uri=config.sagemaker.checkpoint_s3_uri,
            subnets=subnets,
            security_group_ids=security_group_ids,
            environment=environment,
            tags=[
                {"Key": "Project", "Value": config.experiment_name},
            ],
        )

        # Parse objective
        objective = hpo_config.get("objective", {})
        objective_metric = objective.get("metric_name", "eval_loss")
        objective_type = objective.get("type", "Minimize")

        # Parse hyperparameter ranges
        hp_ranges = self._parse_hp_ranges(
            hpo_config.get("hyperparameter_ranges", {}),
            ContinuousParameter,
            IntegerParameter,
            CategoricalParameter,
        )

        strategy = hpo_config.get("strategy", "Bayesian")
        max_jobs = hpo_config.get("max_jobs", 20)
        max_parallel = hpo_config.get("max_parallel_jobs", 4)

        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name=objective_metric,
            objective_type=objective_type,
            hyperparameter_ranges=hp_ranges,
            strategy=strategy,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel,
        )

        # Build input channels
        s3_base = config.output_s3_uri.rstrip("/")
        channels = {
            "train": f"{s3_base}/datasets/{dataset_ref}/train/",
            "validation": f"{s3_base}/datasets/{dataset_ref}/validation/",
        }

        tuner_name = self._generate_job_name(f"hpo-{config.experiment_name}")
        logger.info("Launching HPO tuner", tuner_name=tuner_name)
        tuner.fit(inputs=channels, job_name=tuner_name, wait=False)
        return tuner_name

    # ── private helpers ─────────────────────────────────────────

    @staticmethod
    def _generate_job_name(prefix: str) -> str:
        """Generate a unique SageMaker-compatible job name."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        # SageMaker job names: max 63 chars, alphanumeric + hyphens
        name = f"{prefix}-{ts}-{short_id}"
        # Sanitise: replace non-alphanumeric (except hyphen) with hyphen
        sanitised = "".join(c if c.isalnum() or c == "-" else "-" for c in name)
        return sanitised[:63]

    def _upload_config(self, config: TrainingJobConfig) -> str:
        """Serialise config to YAML and upload to S3.

        Returns:
            The s3:// URI of the uploaded config file.
        """
        config_dict = config.model_dump(mode="json")
        config_yaml = yaml.dump(config_dict, default_flow_style=False)

        s3_key = (
            f"{config.output_s3_uri.rstrip('/').replace('s3://', '').split('/', 1)[1]}/"
            f"configs/{config.run_name or config.experiment_name}/config.yaml"
        )
        bucket = config.output_s3_uri.replace("s3://", "").split("/", 1)[0]

        self._s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=config_yaml.encode("utf-8"),
            ContentType="application/x-yaml",
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId="arn:aws:kms:eu-west-1:162811752071:key/ef340093-c43a-4187-b062-6803dcf99a07",
        )
        uri = f"s3://{bucket}/{s3_key}"
        logger.info("Uploaded config to S3", uri=uri)
        return uri

    def _fetch_cloudwatch_logs(self, job_name: str, tail: int = 50) -> str:
        """Fetch the last *tail* lines from the training job's CloudWatch log."""
        logs_client = boto3.client("logs")
        log_group = "/aws/sagemaker/TrainingJobs"
        try:
            streams = logs_client.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=job_name,
                orderBy="LastEventTime",
                descending=True,
                limit=1,
            )
            if not streams.get("logStreams"):
                return "(no log streams found)"

            stream_name = streams["logStreams"][0]["logStreamName"]
            events_resp = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                startFromHead=False,
                limit=tail,
            )
            lines = [e["message"] for e in events_resp.get("events", [])]
            return "\n".join(lines) if lines else "(no log events)"
        except Exception as exc:
            return f"(failed to fetch logs: {exc})"

    @staticmethod
    def _extract_final_metrics(
        describe_resp: dict[str, Any],
    ) -> dict[str, float]:
        """Extract final metric values from DescribeTrainingJob response."""
        metrics: dict[str, float] = {}
        for metric in describe_resp.get("FinalMetricDataList", []):
            metrics[metric["MetricName"]] = float(metric["Value"])
        return metrics

    @staticmethod
    def _parse_hp_ranges(
        ranges: dict[str, Any],
        continuous_cls: type,
        integer_cls: type,
        categorical_cls: type,
    ) -> dict[str, Any]:
        """Convert raw HPO range dicts to SageMaker parameter objects."""
        parsed: dict[str, Any] = {}
        for name, spec in ranges.items():
            range_type = spec.get("type", "Continuous")
            if range_type == "Continuous":
                kwargs: dict[str, Any] = {
                    "min_value": spec["min"],
                    "max_value": spec["max"],
                }
                if "scaling" in spec:
                    kwargs["scaling_type"] = spec["scaling"]
                parsed[name] = continuous_cls(**kwargs)
            elif range_type == "Integer":
                kwargs = {
                    "min_value": spec["min"],
                    "max_value": spec["max"],
                }
                if "scaling" in spec:
                    kwargs["scaling_type"] = spec["scaling"]
                parsed[name] = integer_cls(**kwargs)
            elif range_type == "Categorical":
                parsed[name] = categorical_cls(spec["values"])
            else:
                raise ValueError(f"Unknown HP range type: {range_type}")
        return parsed
