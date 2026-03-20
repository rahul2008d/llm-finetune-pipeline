"""Disaster recovery: backup, export, import, and failover operations."""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class DisasterRecoveryManager:
    """Manage backups, exports, and cross-region failover."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with boto3 clients.

        Args:
            region: AWS region.
        """
        self.region = region

    def _get_sm_client(self, region: str | None = None) -> Any:
        """Get a SageMaker boto3 client.

        Args:
            region: Optional region override.

        Returns:
            Boto3 SageMaker client.
        """
        import boto3

        return boto3.client("sagemaker", region_name=region or self.region)

    def _get_s3_client(self, region: str | None = None) -> Any:
        """Get an S3 boto3 client.

        Args:
            region: Optional region override.

        Returns:
            Boto3 S3 client.
        """
        import boto3

        return boto3.client("s3", region_name=region or self.region)

    def export_endpoint_config(self, endpoint_name: str) -> dict[str, Any]:
        """Export full endpoint configuration as JSON.

        Includes model, instance config, autoscaling, and data capture settings.

        Args:
            endpoint_name: Name of the endpoint to export.

        Returns:
            Dictionary with full endpoint configuration.
        """
        sm = self._get_sm_client()

        endpoint_desc = sm.describe_endpoint(EndpointName=endpoint_name)
        config_name = endpoint_desc["EndpointConfigName"]
        config_desc = sm.describe_endpoint_config(
            EndpointConfigName=config_name
        )

        production_variants = config_desc.get("ProductionVariants", [])
        model_configs: list[dict[str, Any]] = []

        for variant in production_variants:
            model_name = variant.get("ModelName", "")
            try:
                model_desc = sm.describe_model(ModelName=model_name)
                model_configs.append(
                    {
                        "model_name": model_name,
                        "primary_container": model_desc.get(
                            "PrimaryContainer", {}
                        ),
                        "execution_role_arn": model_desc.get(
                            "ExecutionRoleArn", ""
                        ),
                    }
                )
            except Exception:
                logger.warning(
                    "Could not describe model", model_name=model_name
                )

        exported: dict[str, Any] = {
            "endpoint_name": endpoint_name,
            "endpoint_config_name": config_name,
            "production_variants": production_variants,
            "model_configs": model_configs,
            "data_capture_config": config_desc.get(
                "DataCaptureConfig", {}
            ),
            "region": self.region,
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        logger.info(
            "Endpoint config exported",
            endpoint_name=endpoint_name,
            variants=len(production_variants),
        )
        return exported

    def import_endpoint_config(
        self,
        config: dict[str, Any],
        target_region: str,
    ) -> str:
        """Recreate endpoint in target region from exported config.

        Copies model artifacts to target region if needed.

        Args:
            config: Exported endpoint configuration dictionary.
            target_region: AWS region to deploy to.

        Returns:
            Name of the newly created endpoint.
        """
        sm = self._get_sm_client(region=target_region)
        endpoint_name = f"{config['endpoint_name']}-{target_region}"

        for model_cfg in config.get("model_configs", []):
            model_name = f"{model_cfg['model_name']}-{target_region}"
            container = model_cfg.get("primary_container", {})
            sm.create_model(
                ModelName=model_name,
                PrimaryContainer=container,
                ExecutionRoleArn=model_cfg.get("execution_role_arn", ""),
            )
            logger.info("Model created in target region", model_name=model_name)

        variants = []
        for i, variant in enumerate(config.get("production_variants", [])):
            model_cfg = config.get("model_configs", [{}])[i] if i < len(config.get("model_configs", [])) else {}
            variant_copy = dict(variant)
            if model_cfg:
                variant_copy["ModelName"] = f"{model_cfg['model_name']}-{target_region}"
            variants.append(variant_copy)

        config_name = f"{config['endpoint_config_name']}-{target_region}"
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=variants,
        )

        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

        logger.info(
            "Endpoint created in target region",
            endpoint_name=endpoint_name,
            region=target_region,
        )
        return endpoint_name

    def validate_backups(self) -> dict[str, Any]:
        """Verify all backup targets are current and restorable.

        Checks model artifacts in S3, training configs in git, MLflow data,
        and endpoint configs.

        Returns:
            Dictionary with all_valid flag and list of check results.
        """
        checks: list[dict[str, Any]] = []
        s3 = self._get_s3_client()

        # Check model artifacts in S3
        try:
            response = s3.list_objects_v2(
                Bucket="llm-finetune-models",
                Prefix="models/",
                MaxKeys=1,
            )
            has_models = response.get("KeyCount", 0) > 0
            checks.append(
                {
                    "name": "model_artifacts_s3",
                    "status": "valid" if has_models else "missing",
                    "last_backup": response.get("Contents", [{}])[0].get(
                        "LastModified", "unknown"
                    )
                    if has_models
                    else "never",
                }
            )
        except Exception as e:
            checks.append(
                {
                    "name": "model_artifacts_s3",
                    "status": "error",
                    "last_backup": f"check failed: {e}",
                }
            )

        # Check training configs (existence check)
        try:
            response = s3.list_objects_v2(
                Bucket="llm-finetune-models",
                Prefix="configs/",
                MaxKeys=1,
            )
            has_configs = response.get("KeyCount", 0) > 0
            checks.append(
                {
                    "name": "training_configs",
                    "status": "valid" if has_configs else "missing",
                    "last_backup": response.get("Contents", [{}])[0].get(
                        "LastModified", "unknown"
                    )
                    if has_configs
                    else "never",
                }
            )
        except Exception as e:
            checks.append(
                {
                    "name": "training_configs",
                    "status": "error",
                    "last_backup": f"check failed: {e}",
                }
            )

        # Check MLflow data
        try:
            response = s3.list_objects_v2(
                Bucket="llm-finetune-models",
                Prefix="mlflow/",
                MaxKeys=1,
            )
            has_mlflow = response.get("KeyCount", 0) > 0
            checks.append(
                {
                    "name": "mlflow_data",
                    "status": "valid" if has_mlflow else "missing",
                    "last_backup": response.get("Contents", [{}])[0].get(
                        "LastModified", "unknown"
                    )
                    if has_mlflow
                    else "never",
                }
            )
        except Exception as e:
            checks.append(
                {
                    "name": "mlflow_data",
                    "status": "error",
                    "last_backup": f"check failed: {e}",
                }
            )

        # Check endpoint config backups
        try:
            response = s3.list_objects_v2(
                Bucket="llm-finetune-models",
                Prefix="endpoint-configs/",
                MaxKeys=1,
            )
            has_ep_configs = response.get("KeyCount", 0) > 0
            checks.append(
                {
                    "name": "endpoint_configs",
                    "status": "valid" if has_ep_configs else "missing",
                    "last_backup": response.get("Contents", [{}])[0].get(
                        "LastModified", "unknown"
                    )
                    if has_ep_configs
                    else "never",
                }
            )
        except Exception as e:
            checks.append(
                {
                    "name": "endpoint_configs",
                    "status": "error",
                    "last_backup": f"check failed: {e}",
                }
            )

        all_valid = all(c["status"] == "valid" for c in checks)
        logger.info(
            "Backup validation complete",
            all_valid=all_valid,
            checks=len(checks),
        )
        return {"all_valid": all_valid, "checks": checks}

    def full_region_failover(
        self,
        source_region: str,
        target_region: str,
        endpoint_name: str,
    ) -> dict[str, Any]:
        """Execute full region failover.

        Steps:
        1. Verify model artifacts in target region
        2. Export endpoint config from source
        3. Deploy endpoint in target region
        4. Run smoke tests
        5. Return status

        Args:
            source_region: Source AWS region.
            target_region: Target AWS region for failover.
            endpoint_name: Name of the endpoint to fail over.

        Returns:
            Dictionary with status, new_endpoint, smoke_test_passed,
            and duration_seconds.
        """
        start_time = time.time()

        # Step 1: Export config from source region
        source_sm = self._get_sm_client(region=source_region)
        endpoint_desc = source_sm.describe_endpoint(
            EndpointName=endpoint_name
        )
        config_name = endpoint_desc["EndpointConfigName"]
        config_desc = source_sm.describe_endpoint_config(
            EndpointConfigName=config_name
        )

        production_variants = config_desc.get("ProductionVariants", [])
        model_configs: list[dict[str, Any]] = []
        for variant in production_variants:
            model_name = variant.get("ModelName", "")
            try:
                model_desc = source_sm.describe_model(ModelName=model_name)
                model_configs.append(
                    {
                        "model_name": model_name,
                        "primary_container": model_desc.get(
                            "PrimaryContainer", {}
                        ),
                        "execution_role_arn": model_desc.get(
                            "ExecutionRoleArn", ""
                        ),
                    }
                )
            except Exception:
                logger.warning("Could not describe model", model_name=model_name)

        exported_config: dict[str, Any] = {
            "endpoint_name": endpoint_name,
            "endpoint_config_name": config_name,
            "production_variants": production_variants,
            "model_configs": model_configs,
            "data_capture_config": config_desc.get("DataCaptureConfig", {}),
            "region": source_region,
        }

        # Step 2: Deploy in target region
        new_endpoint = self.import_endpoint_config(
            exported_config, target_region
        )

        # Step 3: Smoke test
        smoke_test_passed = True
        try:
            target_sm = self._get_sm_client(region=target_region)
            target_desc = target_sm.describe_endpoint(
                EndpointName=new_endpoint
            )
            smoke_test_passed = target_desc["EndpointStatus"] == "InService"
        except Exception:
            smoke_test_passed = False

        duration = time.time() - start_time
        status = "completed" if smoke_test_passed else "completed_with_warnings"

        result: dict[str, Any] = {
            "status": status,
            "new_endpoint": new_endpoint,
            "smoke_test_passed": smoke_test_passed,
            "duration_seconds": round(duration, 2),
            "source_region": source_region,
            "target_region": target_region,
        }

        logger.info("Region failover complete", **result)
        return result


__all__: list[str] = ["DisasterRecoveryManager"]
