"""SageMaker container entry point for LLM fine-tuning.

This script runs INSIDE the SageMaker training container.  It reads
environment variables set by SageMaker and the launcher, downloads the
full config from S3, overrides dataset paths, runs training, and writes
outputs to ``SM_MODEL_DIR``.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import boto3
import structlog
import yaml

from src.config.training import TrainingJobConfig
from src.training.trainer import FineTuneTrainer

logger = structlog.get_logger(__name__)


def _parse_sm_env() -> dict[str, str]:
    """Read standard SageMaker environment variables."""
    return {
        "model_dir": os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        "train_dir": os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        "validation_dir": os.environ.get(
            "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation",
        ),
        "output_dir": os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
        "config_s3_uri": os.environ.get("CONFIG_S3_URI", ""),
        "num_gpus": os.environ.get("SM_NUM_GPUS", "1"),
    }


def _download_config(config_s3_uri: str) -> dict[str, Any]:
    """Download the config YAML from S3 and return as a dict."""
    if not config_s3_uri:
        raise ValueError("CONFIG_S3_URI environment variable is not set")

    parts = config_s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    config_dict: dict[str, Any] = yaml.safe_load(body)
    logger.info("Downloaded config from S3", uri=config_s3_uri)
    return config_dict


def _override_dataset_paths(
    config_dict: dict[str, Any],
    train_dir: str,
    validation_dir: str,
) -> dict[str, Any]:
    """Point dataset_path at the local SageMaker input channel directory."""
    config_dict["dataset_path"] = train_dir
    config_dict.pop("dataset_id", None)
    config_dict["_validation_dir"] = validation_dir
    return config_dict


def main() -> None:
    """Entry point executed by SageMaker."""
    logger.info("Starting SageMaker training entry point")

    # 1. Parse SageMaker environment
    sm_env = _parse_sm_env()
    logger.info("SageMaker environment", **sm_env)

    # 2. Download config from S3
    config_dict = _download_config(sm_env["config_s3_uri"])

    # 3. Override dataset paths with SageMaker input channels
    config_dict = _override_dataset_paths(
        config_dict, sm_env["train_dir"], sm_env["validation_dir"],
    )

    # 4. Build validated config and initialise trainer
    config = TrainingJobConfig.model_validate(config_dict)
    trainer = FineTuneTrainer(config)

    # 5. Run training
    logger.info("Starting training")
    result = trainer.train()
    logger.info("Training complete", run_id=result.run_id)

    # 6. Copy adapter to SM_MODEL_DIR
    model_dir = Path(sm_env["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    adapter_source = Path(result.adapter_s3_uri)
    # If the adapter was saved locally (non-S3 path), copy files
    if adapter_source.exists():
        for item in adapter_source.rglob("*"):
            if item.is_file():
                dest = model_dir / item.relative_to(adapter_source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(item), str(dest))
        logger.info("Copied adapter to SM_MODEL_DIR", dest=str(model_dir))

    # 7. Write training_result.json
    result_path = model_dir / "training_result.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
    )
    logger.info("Wrote training result", path=str(result_path))


if __name__ == "__main__":
    main()
