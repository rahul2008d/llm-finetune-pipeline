"""SageMaker Model Registry integration for versioned model management."""

from __future__ import annotations

import json
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]


class ModelRegistryManager:
    """Manage model versions in SageMaker Model Registry."""

    _DEFAULT_INSTANCE_TYPES = [
        "ml.g5.2xlarge",
        "ml.g5.4xlarge",
        "ml.g5.12xlarge",
        "ml.p4d.24xlarge",
    ]

    _DEFAULT_IMAGE_URI = (
        "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
        "huggingface-pytorch-inference:2.1-transformers4.37-gpu-py310-cu121-ubuntu22.04"
    )

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with boto3 SageMaker client.

        Args:
            region: AWS region for SageMaker operations.
        """
        if boto3 is None:
            raise ImportError("boto3 is required for ModelRegistryManager")
        self.region = region
        self._client: Any = boto3.client("sagemaker", region_name=region)

    # ── Registration ────────────────────────────────────────────

    def register_model(
        self,
        model_s3_uri: str,
        model_package_group_name: str,
        training_result: Any,
        eval_results: dict[str, Any] | None = None,
        model_card_content: str | None = None,
        inference_image_uri: str | None = None,
        supported_instance_types: list[str] | None = None,
        approval_status: str = "PendingManualApproval",
    ) -> str:
        """Register a trained model in the registry.

        1. Create ModelPackageGroup if it doesn't exist.
        2. Create ModelPackage with inference spec, metrics, and metadata.
        3. Return model_package_arn.

        Args:
            model_s3_uri: S3 URI of the model artifacts (model.tar.gz).
            model_package_group_name: Name of the model package group.
            training_result: A TrainingResult instance with run metadata.
            eval_results: Optional evaluation metrics dictionary.
            model_card_content: Optional model card markdown content.
            inference_image_uri: Docker image URI for inference container.
            supported_instance_types: Instance types for deployment.
            approval_status: Initial approval status.

        Returns:
            The model package ARN.
        """
        image_uri = inference_image_uri or self._DEFAULT_IMAGE_URI
        instance_types = supported_instance_types or self._DEFAULT_INSTANCE_TYPES

        # 1. Ensure model package group exists
        self._ensure_model_package_group(model_package_group_name)

        # 2. Build CreateModelPackage request
        create_kwargs: dict[str, Any] = {
            "ModelPackageGroupName": model_package_group_name,
            "ModelApprovalStatus": approval_status,
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": image_uri,
                        "ModelDataUrl": model_s3_uri,
                    }
                ],
                "SupportedContentTypes": ["application/json"],
                "SupportedResponseMIMETypes": ["application/json"],
                "SupportedRealtimeInferenceInstanceTypes": instance_types,
            },
        }

        # Customer metadata
        metadata: dict[str, str] = {
            "run_id": str(getattr(training_result, "run_id", "")),
            "experiment_name": str(getattr(training_result, "experiment_name", "")),
            "final_train_loss": str(getattr(training_result, "final_train_loss", "")),
            "best_eval_loss": str(getattr(training_result, "best_eval_loss", "")),
            "training_time_seconds": str(
                getattr(training_result, "training_time_seconds", "")
            ),
        }
        if model_card_content:
            # Truncate to fit 256-char value limit per key
            metadata["model_card_summary"] = model_card_content[:256]

        create_kwargs["CustomerMetadataProperties"] = metadata

        # Model metrics (eval results)
        if eval_results:
            metrics_body = json.dumps(eval_results)
            create_kwargs["ModelMetrics"] = {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "Body": metrics_body,
                    }
                }
            }

        # 3. Create model package
        response = self._client.create_model_package(**create_kwargs)
        model_package_arn: str = response["ModelPackageArn"]

        logger.info(
            "Model registered",
            model_package_arn=model_package_arn,
            group=model_package_group_name,
            status=approval_status,
        )
        return model_package_arn

    # ── Approval Management ────────────────────────────────────

    def approve_model(self, model_package_arn: str) -> None:
        """Update approval status to 'Approved'.

        Args:
            model_package_arn: ARN of the model package to approve.
        """
        self._client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Approved",
        )
        logger.info("Model approved", model_package_arn=model_package_arn)

    def reject_model(self, model_package_arn: str, reason: str) -> None:
        """Update approval status to 'Rejected' with reason.

        Args:
            model_package_arn: ARN of the model package to reject.
            reason: Reason for rejection.
        """
        self._client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Rejected",
            ApprovalDescription=reason,
        )
        logger.info(
            "Model rejected",
            model_package_arn=model_package_arn,
            reason=reason,
        )

    # ── Queries ─────────────────────────────────────────────────

    def get_latest_approved(self, group_name: str) -> dict[str, Any]:
        """Query for latest model with Approved status.

        Args:
            group_name: Name of the model package group.

        Returns:
            Dict with model_package_arn, model_data_url, creation_time,
            and metrics. Empty dict if no approved model found.
        """
        response = self._client.list_model_packages(
            ModelPackageGroupName=group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )

        packages = response.get("ModelPackageSummaryList", [])
        if not packages:
            logger.info("No approved model found", group=group_name)
            return {}

        pkg = packages[0]
        arn = pkg["ModelPackageArn"]

        # Get full details for model_data_url and metrics
        details = self._client.describe_model_package(ModelPackageName=arn)

        model_data_url = ""
        containers = (
            details.get("InferenceSpecification", {}).get("Containers", [])
        )
        if containers:
            model_data_url = containers[0].get("ModelDataUrl", "")

        metrics: dict[str, Any] = {}
        model_quality = details.get("ModelMetrics", {}).get("ModelQuality", {})
        stats_body = model_quality.get("Statistics", {}).get("Body", "")
        if stats_body:
            try:
                metrics = json.loads(stats_body)
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "model_package_arn": arn,
            "model_data_url": model_data_url,
            "creation_time": str(pkg.get("CreationTime", "")),
            "metrics": metrics,
        }

    def list_versions(self, group_name: str) -> list[dict[str, Any]]:
        """List all model versions with status and metrics.

        Args:
            group_name: Name of the model package group.

        Returns:
            List of dicts with version info.
        """
        versions: list[dict[str, Any]] = []
        paginator = self._client.get_paginator("list_model_packages")

        for page in paginator.paginate(ModelPackageGroupName=group_name):
            for pkg in page.get("ModelPackageSummaryList", []):
                versions.append(
                    {
                        "model_package_arn": pkg["ModelPackageArn"],
                        "status": pkg.get("ModelApprovalStatus", "Unknown"),
                        "creation_time": str(pkg.get("CreationTime", "")),
                    }
                )

        logger.info("Listed model versions", group=group_name, count=len(versions))
        return versions

    def get_model_lineage(self, model_package_arn: str) -> dict[str, Any]:
        """Return lineage info: training config, dataset_id, code commit, eval results.

        Args:
            model_package_arn: ARN of the model package.

        Returns:
            Dict with lineage information extracted from metadata and metrics.
        """
        details = self._client.describe_model_package(
            ModelPackageName=model_package_arn
        )

        metadata = details.get("CustomerMetadataProperties", {})

        # Extract eval results from model metrics
        eval_results: dict[str, Any] = {}
        model_quality = details.get("ModelMetrics", {}).get("ModelQuality", {})
        stats_body = model_quality.get("Statistics", {}).get("Body", "")
        if stats_body:
            try:
                eval_results = json.loads(stats_body)
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "run_id": metadata.get("run_id", ""),
            "experiment_name": metadata.get("experiment_name", ""),
            "final_train_loss": metadata.get("final_train_loss", ""),
            "best_eval_loss": metadata.get("best_eval_loss", ""),
            "training_time_seconds": metadata.get("training_time_seconds", ""),
            "model_card_summary": metadata.get("model_card_summary", ""),
            "eval_results": eval_results,
        }

    # ── Private helpers ─────────────────────────────────────────

    def _ensure_model_package_group(self, group_name: str) -> None:
        """Create model package group if it doesn't exist.

        Args:
            group_name: Name of the model package group.
        """
        try:
            self._client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            logger.debug("Model package group exists", group=group_name)
        except self._client.exceptions.ClientError:
            self._client.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=f"Fine-tuned models: {group_name}",
            )
            logger.info("Created model package group", group=group_name)


__all__: list[str] = ["ModelRegistryManager"]
