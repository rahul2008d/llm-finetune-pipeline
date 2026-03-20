"""SageMaker real-time endpoint management with blue-green deployment."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]


_DEFAULT_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.1-transformers4.37-gpu-py310-cu121-ubuntu22.04"
)


class SageMakerEndpointManager:
    """Create, update, and manage SageMaker real-time endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with boto3 SageMaker client.

        Args:
            region: AWS region for SageMaker operations.
        """
        if boto3 is None:
            raise ImportError("boto3 is required for SageMakerEndpointManager")
        self.region = region
        self._client: Any = boto3.client("sagemaker", region_name=region)

    def create_endpoint(
        self,
        model_package_arn: str | None = None,
        model_data_url: str | None = None,
        endpoint_name: str = "",
        instance_type: str = "ml.g5.xlarge",
        initial_instance_count: int = 1,
        role_arn: str = "",
        container_image: str | None = None,
        data_capture_enabled: bool = False,
        data_capture_s3_uri: str = "",
        data_capture_sampling_pct: int = 100,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a new SageMaker endpoint.

        1. Create Model (from model_package_arn or model_data_url + container).
        2. Create EndpointConfig with ProductionVariant and optional DataCaptureConfig.
        3. Create Endpoint.
        4. Wait for InService status (timeout 15 minutes).
        5. Return dict with endpoint_name, endpoint_arn, status, url.

        Args:
            model_package_arn: ARN of a registered model package.
            model_data_url: S3 URI of model.tar.gz (alternative to model_package_arn).
            endpoint_name: Name for the endpoint (auto-generated if empty).
            instance_type: EC2 instance type for hosting.
            initial_instance_count: Number of instances.
            role_arn: IAM role ARN for SageMaker.
            container_image: Docker image URI for inference.
            data_capture_enabled: Whether to enable data capture.
            data_capture_s3_uri: S3 URI for captured data.
            data_capture_sampling_pct: Percentage of requests to capture.
            tags: Resource tags.

        Returns:
            Dict with endpoint_name, endpoint_arn, status.
        """
        if not endpoint_name:
            endpoint_name = f"llm-ft-{uuid.uuid4().hex[:8]}"

        model_name = f"{endpoint_name}-model"
        config_name = f"{endpoint_name}-config"
        variant_name = "AllTraffic"
        image_uri = container_image or _DEFAULT_IMAGE_URI
        aws_tags = [{"Key": k, "Value": v} for k, v in (tags or {}).items()]

        # 1. Create Model
        model_kwargs: dict[str, Any] = {
            "ModelName": model_name,
            "ExecutionRoleArn": role_arn,
        }
        if model_package_arn:
            model_kwargs["Containers"] = [
                {"ModelPackageName": model_package_arn}
            ]
        else:
            model_kwargs["PrimaryContainer"] = {
                "Image": image_uri,
                "ModelDataUrl": model_data_url or "",
            }
        if aws_tags:
            model_kwargs["Tags"] = aws_tags

        self._client.create_model(**model_kwargs)
        logger.info("Created model", model_name=model_name)

        # 2. Create EndpointConfig
        config_kwargs: dict[str, Any] = {
            "EndpointConfigName": config_name,
            "ProductionVariants": [
                {
                    "VariantName": variant_name,
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": initial_instance_count,
                    "InitialVariantWeight": 1.0,
                }
            ],
        }
        if data_capture_enabled:
            config_kwargs["DataCaptureConfig"] = {
                "EnableCapture": True,
                "InitialSamplingPercentage": data_capture_sampling_pct,
                "DestinationS3Uri": data_capture_s3_uri,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
            }
        if aws_tags:
            config_kwargs["Tags"] = aws_tags

        self._client.create_endpoint_config(**config_kwargs)
        logger.info("Created endpoint config", config_name=config_name)

        # 3. Create Endpoint
        endpoint_kwargs: dict[str, Any] = {
            "EndpointName": endpoint_name,
            "EndpointConfigName": config_name,
        }
        if aws_tags:
            endpoint_kwargs["Tags"] = aws_tags

        self._client.create_endpoint(**endpoint_kwargs)
        logger.info("Creating endpoint", endpoint_name=endpoint_name)

        # 4. Wait for InService
        self._wait_for_endpoint(endpoint_name, timeout_minutes=15)

        # 5. Describe and return
        desc = self._client.describe_endpoint(EndpointName=endpoint_name)
        return {
            "endpoint_name": endpoint_name,
            "endpoint_arn": desc.get("EndpointArn", ""),
            "status": desc.get("EndpointStatus", ""),
        }

    def update_endpoint_traffic(
        self,
        endpoint_name: str,
        target_variant: str,
        target_weight: int,
    ) -> None:
        """Shift traffic between variants for canary/A-B deployment.

        Args:
            endpoint_name: Name of the endpoint.
            target_variant: Variant to update.
            target_weight: New weight (0-100).
        """
        self._client.update_endpoint_weights_and_capacities(
            EndpointName=endpoint_name,
            DesiredWeightsAndCapacities=[
                {
                    "VariantName": target_variant,
                    "DesiredWeight": float(target_weight),
                }
            ],
        )
        logger.info(
            "Updated traffic",
            endpoint=endpoint_name,
            variant=target_variant,
            weight=target_weight,
        )

    def blue_green_deploy(
        self,
        endpoint_name: str,
        new_model_data_url: str,
        new_model_package_arn: str | None = None,
        instance_type: str = "ml.g5.xlarge",
        canary_pct: float = 0.1,
        bake_time_minutes: int = 30,
        rollback_alarm_names: list[str] | None = None,
        role_arn: str = "",
        container_image: str | None = None,
    ) -> dict[str, Any]:
        """Blue-green deployment with canary traffic shifting.

        1. Create new model variant with new model.
        2. Update endpoint config to add new variant at canary_pct weight.
        3. Apply the update to the endpoint.
        4. Monitor for bake_time_minutes (poll CloudWatch alarms if provided).
        5. If stable -> shift 100% to new variant, remove old variant.
        6. If alarm fires -> rollback.

        Args:
            endpoint_name: Existing endpoint name.
            new_model_data_url: S3 URI of new model artifacts.
            new_model_package_arn: Optional model package ARN.
            instance_type: Instance type for new variant.
            canary_pct: Initial traffic percentage for new variant.
            bake_time_minutes: Monitoring duration before full cutover.
            rollback_alarm_names: CloudWatch alarm names for auto-rollback.
            role_arn: IAM role ARN.
            container_image: Docker image URI.

        Returns:
            Dict with status, old_variant, new_variant, rolled_back flag.
        """
        image_uri = container_image or _DEFAULT_IMAGE_URI
        new_variant_name = f"new-{uuid.uuid4().hex[:6]}"
        new_model_name = f"{endpoint_name}-model-{uuid.uuid4().hex[:6]}"

        # 1. Create new model
        model_kwargs: dict[str, Any] = {
            "ModelName": new_model_name,
            "ExecutionRoleArn": role_arn,
        }
        if new_model_package_arn:
            model_kwargs["Containers"] = [
                {"ModelPackageName": new_model_package_arn}
            ]
        else:
            model_kwargs["PrimaryContainer"] = {
                "Image": image_uri,
                "ModelDataUrl": new_model_data_url,
            }
        self._client.create_model(**model_kwargs)

        # 2. Get current endpoint config to find old variant
        desc = self._client.describe_endpoint(EndpointName=endpoint_name)
        old_config_name = desc["EndpointConfigName"]
        old_config = self._client.describe_endpoint_config(
            EndpointConfigName=old_config_name
        )
        old_variants = old_config["ProductionVariants"]
        old_variant_name = old_variants[0]["VariantName"]

        # 3. Create new endpoint config with both variants
        new_config_name = f"{endpoint_name}-bg-{uuid.uuid4().hex[:6]}"
        canary_weight = canary_pct
        old_weight = 1.0 - canary_pct

        new_variants = [
            {
                "VariantName": old_variant_name,
                "ModelName": old_variants[0]["ModelName"],
                "InstanceType": old_variants[0].get("InstanceType", instance_type),
                "InitialInstanceCount": old_variants[0].get(
                    "InitialInstanceCount", 1
                ),
                "InitialVariantWeight": old_weight,
            },
            {
                "VariantName": new_variant_name,
                "ModelName": new_model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
                "InitialVariantWeight": canary_weight,
            },
        ]

        self._client.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=new_variants,
        )

        # 4. Apply update
        self._client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=new_config_name,
        )
        self._wait_for_endpoint(endpoint_name, timeout_minutes=15)

        logger.info(
            "Canary deployed",
            endpoint=endpoint_name,
            new_variant=new_variant_name,
            canary_pct=canary_pct,
        )

        # 5. Monitor during bake time
        rolled_back = False
        if rollback_alarm_names:
            rolled_back = self._monitor_alarms(
                alarm_names=rollback_alarm_names,
                duration_minutes=bake_time_minutes,
                poll_interval_seconds=30,
            )

        if rolled_back:
            # Rollback: shift all traffic to old variant
            self.update_endpoint_traffic(endpoint_name, old_variant_name, 1)
            self.update_endpoint_traffic(endpoint_name, new_variant_name, 0)
            logger.warning(
                "Rolled back deployment",
                endpoint=endpoint_name,
                new_variant=new_variant_name,
            )
            return {
                "status": "rolled_back",
                "old_variant": old_variant_name,
                "new_variant": new_variant_name,
                "rolled_back": True,
            }

        # 6. Shift 100% to new variant
        self.update_endpoint_traffic(endpoint_name, new_variant_name, 1)
        self.update_endpoint_traffic(endpoint_name, old_variant_name, 0)

        logger.info(
            "Blue-green deployment complete",
            endpoint=endpoint_name,
            new_variant=new_variant_name,
        )

        return {
            "status": "completed",
            "old_variant": old_variant_name,
            "new_variant": new_variant_name,
            "rolled_back": False,
        }

    def delete_endpoint(
        self, endpoint_name: str, delete_model: bool = True
    ) -> None:
        """Delete endpoint, endpoint config, and optionally the model.

        Args:
            endpoint_name: Name of the endpoint to delete.
            delete_model: Whether to also delete the underlying model.
        """
        # Get config and model names before deletion
        model_name = None
        config_name = None
        try:
            desc = self._client.describe_endpoint(EndpointName=endpoint_name)
            config_name = desc.get("EndpointConfigName")
            if config_name and delete_model:
                config_desc = self._client.describe_endpoint_config(
                    EndpointConfigName=config_name
                )
                variants = config_desc.get("ProductionVariants", [])
                if variants:
                    model_name = variants[0].get("ModelName")
        except Exception:
            logger.warning("Could not describe endpoint before deletion")

        self._client.delete_endpoint(EndpointName=endpoint_name)
        logger.info("Deleted endpoint", endpoint_name=endpoint_name)

        if config_name:
            try:
                self._client.delete_endpoint_config(
                    EndpointConfigName=config_name
                )
                logger.info("Deleted endpoint config", config_name=config_name)
            except Exception:
                logger.warning("Could not delete endpoint config")

        if delete_model and model_name:
            try:
                self._client.delete_model(ModelName=model_name)
                logger.info("Deleted model", model_name=model_name)
            except Exception:
                logger.warning("Could not delete model")

    def describe_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        """Full endpoint status, variants, traffic distribution.

        Args:
            endpoint_name: Name of the endpoint.

        Returns:
            Dict with status, endpoint_arn, variants list, timestamps.
        """
        desc = self._client.describe_endpoint(EndpointName=endpoint_name)

        variants: list[dict[str, Any]] = []
        for v in desc.get("ProductionVariants", []):
            variants.append(
                {
                    "name": v.get("VariantName", ""),
                    "instance_type": v.get("CurrentInstanceCount", 0),
                    "weight": v.get("CurrentWeight", 0),
                    "status": v.get("VariantStatus", [{}])[0].get("Status", "")
                    if v.get("VariantStatus")
                    else "",
                }
            )

        return {
            "status": desc.get("EndpointStatus", ""),
            "endpoint_arn": desc.get("EndpointArn", ""),
            "variants": variants,
            "creation_time": str(desc.get("CreationTime", "")),
            "last_modified": str(desc.get("LastModifiedTime", "")),
        }

    def list_endpoints(self, name_contains: str = "") -> list[dict[str, Any]]:
        """List endpoints, optionally filtered by name prefix.

        Args:
            name_contains: Filter by name prefix.

        Returns:
            List of endpoint summary dicts.
        """
        kwargs: dict[str, Any] = {}
        if name_contains:
            kwargs["NameContains"] = name_contains

        response = self._client.list_endpoints(**kwargs)
        endpoints: list[dict[str, Any]] = []
        for ep in response.get("Endpoints", []):
            endpoints.append(
                {
                    "endpoint_name": ep.get("EndpointName", ""),
                    "status": ep.get("EndpointStatus", ""),
                    "creation_time": str(ep.get("CreationTime", "")),
                }
            )
        return endpoints

    # ── Private helpers ─────────────────────────────────────────

    def _wait_for_endpoint(
        self, endpoint_name: str, timeout_minutes: int = 15
    ) -> None:
        """Poll endpoint until InService or timeout.

        Args:
            endpoint_name: Endpoint to wait on.
            timeout_minutes: Maximum wait time.

        Raises:
            TimeoutError: If endpoint doesn't become InService.
            RuntimeError: If endpoint enters a Failed state.
        """
        deadline = time.time() + timeout_minutes * 60
        while time.time() < deadline:
            desc = self._client.describe_endpoint(EndpointName=endpoint_name)
            status = desc.get("EndpointStatus", "")
            if status == "InService":
                logger.info("Endpoint is InService", endpoint=endpoint_name)
                return
            if status == "Failed":
                reason = desc.get("FailureReason", "Unknown")
                raise RuntimeError(
                    f"Endpoint {endpoint_name} failed: {reason}"
                )
            time.sleep(30)

        raise TimeoutError(
            f"Endpoint {endpoint_name} not InService after {timeout_minutes}m"
        )

    def _monitor_alarms(
        self,
        alarm_names: list[str],
        duration_minutes: int,
        poll_interval_seconds: int = 30,
    ) -> bool:
        """Monitor CloudWatch alarms and return True if any fires.

        Args:
            alarm_names: CloudWatch alarm names to monitor.
            duration_minutes: How long to monitor.
            poll_interval_seconds: Polling interval.

        Returns:
            True if any alarm fired (should rollback), False if stable.
        """
        cw_client = boto3.client("cloudwatch", region_name=self.region)
        deadline = time.time() + duration_minutes * 60

        while time.time() < deadline:
            response = cw_client.describe_alarms(AlarmNames=alarm_names)
            for alarm in response.get("MetricAlarms", []):
                if alarm.get("StateValue") == "ALARM":
                    logger.warning(
                        "Alarm triggered",
                        alarm=alarm["AlarmName"],
                    )
                    return True
            time.sleep(poll_interval_seconds)

        return False


class SageMakerEndpointHandler:
    """Handle SageMaker endpoint deployment and inference requests."""

    def __init__(
        self,
        model_path: str,
        endpoint_name: str,
        instance_type: str = "ml.g5.2xlarge",
        instance_count: int = 1,
    ) -> None:
        """Initialize the SageMaker endpoint handler.

        Args:
            model_path: S3 URI or local path to the model artifacts.
            endpoint_name: Name for the SageMaker endpoint.
            instance_type: EC2 instance type for the endpoint.
            instance_count: Number of instances to deploy.
        """
        self.model_path = model_path
        self.endpoint_name = endpoint_name
        self.instance_type = instance_type
        self.instance_count = instance_count

    def deploy(self, role_arn: str, wait: bool = True) -> str:
        """Deploy the model to a SageMaker endpoint.

        Args:
            role_arn: IAM role ARN for SageMaker execution.
            wait: Whether to wait for deployment to complete.

        Returns:
            The endpoint name.
        """
        import sagemaker
        from sagemaker.huggingface import HuggingFaceModel

        logger.info(
            "Deploying to SageMaker",
            endpoint=self.endpoint_name,
            instance_type=self.instance_type,
        )

        sess = sagemaker.Session()
        model = HuggingFaceModel(
            model_data=self.model_path,
            role=role_arn,
            transformers_version="4.37",
            pytorch_version="2.1",
            py_version="py310",
            sagemaker_session=sess,
        )

        model.deploy(
            initial_instance_count=self.instance_count,
            instance_type=self.instance_type,
            endpoint_name=self.endpoint_name,
            wait=wait,
        )

        logger.info("Deployment complete", endpoint=self.endpoint_name)
        return self.endpoint_name

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send an inference request to the deployed endpoint.

        Args:
            payload: Input payload with prompt and generation parameters.

        Returns:
            Model response dictionary.
        """
        import boto3

        client: Any = boto3.client("sagemaker-runtime")
        import json

        response = client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result: dict[str, Any] = json.loads(response["Body"].read().decode())
        return result

    def delete(self) -> None:
        """Delete the SageMaker endpoint and associated resources."""
        import boto3

        client: Any = boto3.client("sagemaker")
        logger.info("Deleting endpoint", endpoint=self.endpoint_name)

        client.delete_endpoint(EndpointName=self.endpoint_name)
        client.delete_endpoint_config(EndpointConfigName=self.endpoint_name)
        logger.info("Endpoint deleted")


__all__: list[str] = ["SageMakerEndpointHandler", "SageMakerEndpointManager"]
