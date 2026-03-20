"""AWS Bedrock Custom Model Import and provisioned throughput management."""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

_SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "PhiForCausalLM",
}

_MAX_MODEL_SIZE_GB = 50.0


class BedrockImportManager:
    """Import custom models into Bedrock and manage provisioned throughput."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with bedrock and bedrock-runtime clients.

        Args:
            region: AWS region for Bedrock operations.
        """
        if boto3 is None:
            raise ImportError("boto3 is required for BedrockImportManager")
        self.region = region
        self._client: Any = boto3.client("bedrock", region_name=region)
        self._runtime_client: Any = boto3.client(
            "bedrock-runtime", region_name=region
        )

    def import_model(
        self,
        model_name: str,
        model_s3_uri: str,
        role_arn: str,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Import a model into Bedrock.

        1. Validate model artifacts at S3 URI.
        2. Call create_model_import_job.
        3. Poll get_model_import_job until Complete or Failed (timeout 60 min).
        4. On failure: fetch error details, log, raise.
        5. Return: {model_arn, import_job_arn, status}.

        Args:
            model_name: Name for the imported model.
            model_s3_uri: S3 URI of the model artifacts.
            role_arn: IAM role ARN with Bedrock permissions.
            tags: Optional resource tags.

        Returns:
            Dict with model_arn, import_job_arn, status.

        Raises:
            ValueError: If model artifacts fail validation.
            RuntimeError: If import job fails.
        """
        # 1. Validate artifacts
        self._validate_s3_artifacts(model_s3_uri)

        # 2. Create import job
        job_name = f"{model_name}-import-{int(time.time())}"
        create_kwargs: dict[str, Any] = {
            "jobName": job_name,
            "importedModelName": model_name,
            "roleArn": role_arn,
            "modelDataSource": {
                "s3DataSource": {
                    "s3Uri": model_s3_uri,
                }
            },
        }
        if tags:
            create_kwargs["tags"] = [
                {"key": k, "value": v} for k, v in tags.items()
            ]

        response = self._client.create_model_import_job(**create_kwargs)
        job_arn: str = response["jobArn"]
        logger.info("Created import job", job_arn=job_arn, model=model_name)

        # 3. Poll until complete (timeout 60 min)
        status = self._poll_import_job(job_arn, timeout_minutes=60)

        if status.get("status") == "Failed":
            error = status.get("failureMessage", "Unknown error")
            raise RuntimeError(f"Bedrock import failed: {error}")

        model_arn = status.get("importedModelArn", "")

        logger.info(
            "Model imported",
            model_arn=model_arn,
            status=status.get("status"),
        )

        return {
            "model_arn": model_arn,
            "import_job_arn": job_arn,
            "status": status.get("status", ""),
        }

    def create_provisioned_throughput(
        self,
        model_arn: str,
        throughput_name: str,
        model_units: int = 1,
        commitment: str = "NO_COMMITMENT",
    ) -> dict[str, Any]:
        """Create provisioned throughput for the imported model.

        Waits for Active status (timeout 30 min).

        Args:
            model_arn: ARN of the imported model.
            throughput_name: Name for the provisioned throughput.
            model_units: Number of model units.
            commitment: Commitment type (NO_COMMITMENT, ONE_MONTH, SIX_MONTH).

        Returns:
            Dict with provisioned_model_arn and status.
        """
        response = self._client.create_provisioned_model_throughput(
            modelUnits=model_units,
            provisionedModelName=throughput_name,
            modelId=model_arn,
            commitmentDuration=commitment,
        )
        provisioned_arn: str = response["provisionedModelArn"]

        logger.info(
            "Creating provisioned throughput",
            arn=provisioned_arn,
            units=model_units,
        )

        # Wait for Active
        deadline = time.time() + 30 * 60
        while time.time() < deadline:
            desc = self._client.get_provisioned_model_throughput(
                provisionedModelId=provisioned_arn
            )
            status = desc.get("status", "")
            if status == "InService":
                break
            if status == "Failed":
                raise RuntimeError(
                    f"Provisioned throughput failed: {desc.get('failureMessage', '')}"
                )
            time.sleep(30)

        return {
            "provisioned_model_arn": provisioned_arn,
            "status": status,
        }

    def invoke_model(
        self,
        provisioned_model_arn: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        """Invoke the model via bedrock-runtime.

        Args:
            provisioned_model_arn: ARN of the provisioned model.
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.

        Returns:
            Dict with generated_text, input_tokens, output_tokens, latency_ms.
        """
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

        start = time.perf_counter()
        response = self._runtime_client.invoke_model(
            modelId=provisioned_model_arn,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        latency_ms = (time.perf_counter() - start) * 1000

        result = json.loads(response["body"].read())

        return {
            "generated_text": result.get("generation", ""),
            "input_tokens": result.get(
                "prompt_token_count", 0
            ),
            "output_tokens": result.get(
                "generation_token_count", 0
            ),
            "latency_ms": round(latency_ms, 2),
        }

    def delete_model(self, model_arn: str) -> None:
        """Delete provisioned throughput first (if any), then delete the model.

        Args:
            model_arn: ARN of the model to delete.
        """
        # Check for provisioned throughput
        try:
            response = self._client.list_provisioned_model_throughputs()
            for pt in response.get("provisionedModelSummaries", []):
                if pt.get("modelArn") == model_arn:
                    self._client.delete_provisioned_model_throughput(
                        provisionedModelId=pt["provisionedModelArn"]
                    )
                    logger.info(
                        "Deleted provisioned throughput",
                        arn=pt["provisionedModelArn"],
                    )
        except Exception:
            logger.debug("No provisioned throughput to clean up")

        self._client.delete_imported_model(modelIdentifier=model_arn)
        logger.info("Deleted model", model_arn=model_arn)

    def list_custom_models(self) -> list[dict[str, Any]]:
        """List all custom imported models.

        Returns:
            List of model summary dicts.
        """
        response = self._client.list_imported_models()
        models: list[dict[str, Any]] = response.get("modelSummaries", [])
        logger.info("Listed custom models", count=len(models))
        return models

    # ── Private helpers ─────────────────────────────────────────

    def _validate_s3_artifacts(self, model_s3_uri: str) -> None:
        """Validate model artifacts at S3 URI.

        Args:
            model_s3_uri: S3 URI of model artifacts.

        Raises:
            ValueError: If validation fails.
        """
        s3_client: Any = boto3.client("s3", region_name=self.region)
        bucket, prefix = self._parse_s3_uri(model_s3_uri)

        # List objects
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = response.get("Contents", [])
        if not objects:
            raise ValueError(f"No objects found at {model_s3_uri}")

        filenames = {obj["Key"].split("/")[-1] for obj in objects}
        total_size = sum(obj.get("Size", 0) for obj in objects)
        size_gb = total_size / (1024**3)

        # Check config.json
        if "config.json" not in filenames:
            raise ValueError("Missing config.json in model artifacts")

        # Validate architecture by downloading config.json
        config_key = f"{prefix}/config.json" if prefix else "config.json"
        # Find the actual config key
        for obj in objects:
            if obj["Key"].endswith("config.json"):
                config_key = obj["Key"]
                break

        config_resp = s3_client.get_object(Bucket=bucket, Key=config_key)
        config_data = json.loads(config_resp["Body"].read())
        architectures = config_data.get("architectures", [])

        if not architectures:
            raise ValueError("config.json missing 'architectures' field")
        if architectures[0] not in _SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architectures[0]}. "
                f"Supported: {_SUPPORTED_ARCHITECTURES}"
            )

        # Check tokenizer
        required_tokenizer = {"tokenizer.json", "tokenizer_config.json"}
        missing = required_tokenizer - filenames
        if missing:
            raise ValueError(f"Missing tokenizer files: {missing}")

        # Check safetensors
        has_safetensors = any(f.endswith(".safetensors") for f in filenames)
        if not has_safetensors:
            raise ValueError("No safetensors files found")

        # Size check
        if size_gb > _MAX_MODEL_SIZE_GB:
            raise ValueError(
                f"Model size {size_gb:.1f}GB exceeds limit of {_MAX_MODEL_SIZE_GB}GB"
            )

        logger.info(
            "Artifacts validated",
            architecture=architectures[0],
            size_gb=round(size_gb, 2),
        )

    def _poll_import_job(
        self, job_arn: str, timeout_minutes: int = 60
    ) -> dict[str, Any]:
        """Poll import job until terminal state.

        Args:
            job_arn: ARN of the import job.
            timeout_minutes: Max wait time.

        Returns:
            Final job status dict.
        """
        deadline = time.time() + timeout_minutes * 60
        while time.time() < deadline:
            status = self._client.get_model_import_job(jobIdentifier=job_arn)
            job_status = status.get("status", "")
            if job_status in ("Completed", "Failed"):
                return status
            logger.debug("Import job polling", status=job_status)
            time.sleep(30)

        return {"status": "Timeout", "failureMessage": "Import timed out"}

    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and prefix.

        Args:
            s3_uri: S3 URI.

        Returns:
            Tuple of (bucket, prefix).
        """
        path = s3_uri.replace("s3://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
        return bucket, prefix


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


__all__: list[str] = ["BedrockImporter", "BedrockImportManager"]
