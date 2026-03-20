"""SageMaker endpoint handler for model inference."""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


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


__all__: list[str] = ["SageMakerEndpointHandler"]
