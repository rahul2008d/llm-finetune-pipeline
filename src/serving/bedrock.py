"""Bedrock model import scripts for deploying fine-tuned models to AWS Bedrock."""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BedrockImporter:
    """Import fine-tuned models into AWS Bedrock."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize the Bedrock importer.

        Args:
            region: AWS region for Bedrock operations.
        """
        self.region = region

    def create_model_import_job(
        self,
        job_name: str,
        model_name: str,
        model_data_url: str,
        role_arn: str,
    ) -> str:
        """Create a Bedrock custom model import job.

        Args:
            job_name: Name for the import job.
            model_name: Name for the imported model.
            model_data_url: S3 URI of the model artifacts.
            role_arn: IAM role ARN with Bedrock permissions.

        Returns:
            The import job ARN.
        """
        import boto3

        client: Any = boto3.client("bedrock", region_name=self.region)

        logger.info(
            "Creating Bedrock model import job",
            job_name=job_name,
            model_name=model_name,
            model_data_url=model_data_url,
        )

        response = client.create_model_import_job(
            jobName=job_name,
            importedModelName=model_name,
            roleArn=role_arn,
            modelDataSource={
                "s3DataSource": {
                    "s3Uri": model_data_url,
                }
            },
        )

        job_arn: str = response["jobArn"]
        logger.info("Import job created", job_arn=job_arn)
        return job_arn

    def get_import_job_status(self, job_arn: str) -> dict[str, Any]:
        """Check the status of a model import job.

        Args:
            job_arn: ARN of the import job.

        Returns:
            Job status information.
        """
        import boto3

        client: Any = boto3.client("bedrock", region_name=self.region)
        response: dict[str, Any] = client.get_model_import_job(jobIdentifier=job_arn)
        logger.info("Import job status", status=response.get("status"))
        return response

    def list_imported_models(self) -> list[dict[str, Any]]:
        """List all imported custom models in Bedrock.

        Returns:
            List of imported model summaries.
        """
        import boto3

        client: Any = boto3.client("bedrock", region_name=self.region)
        response = client.list_imported_models()
        models: list[dict[str, Any]] = response.get("modelSummaries", [])
        logger.info("Listed imported models", count=len(models))
        return models


__all__: list[str] = ["BedrockImporter"]
