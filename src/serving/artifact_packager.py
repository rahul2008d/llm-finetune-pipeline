"""Package model artifacts for SageMaker and Bedrock deployment."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

# Bedrock-supported architectures
_SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "CohereForCausalLM",
}

# Required tokenizer files
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}

_BEDROCK_MAX_SIZE_GB = 50.0


class ArtifactPackager:
    """Package model artifacts for different deployment targets."""

    def package_for_sagemaker(
        self,
        model_path: str,
        output_path: str,
        inference_code_path: str = "src/serving/inference.py",
    ) -> str:
        """Create model.tar.gz for SageMaker deployment.

        Contents:
          model/  (safetensors weights, config.json, tokenizer files)
          code/inference.py  (SageMaker inference handler)
          code/requirements.txt  (serving dependencies)

        Upload to S3 if output_path starts with s3://.

        Args:
            model_path: Local path to model directory.
            output_path: Local path or S3 URI for the output archive.
            inference_code_path: Path to inference handler script.

        Returns:
            The output path (local or S3 URI).
        """
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tarball_path = os.path.join(tmpdir, "model.tar.gz")

            with tarfile.open(tarball_path, "w:gz") as tar:
                # Add model files under model/
                for fpath in model_dir.rglob("*"):
                    if fpath.is_file():
                        arcname = f"model/{fpath.relative_to(model_dir)}"
                        tar.add(str(fpath), arcname=arcname)

                # Add inference code under code/
                inference_file = Path(inference_code_path)
                if inference_file.is_file():
                    tar.add(str(inference_file), arcname="code/inference.py")
                else:
                    logger.warning(
                        "Inference code not found",
                        path=inference_code_path,
                    )

                # Create and add requirements.txt under code/
                reqs_path = os.path.join(tmpdir, "requirements.txt")
                with open(reqs_path, "w") as f:
                    f.write(
                        "torch>=2.0\n"
                        "transformers>=4.37\n"
                        "accelerate>=0.25\n"
                        "safetensors>=0.4\n"
                        "peft>=0.7\n"
                    )
                tar.add(reqs_path, arcname="code/requirements.txt")

            if output_path.startswith("s3://"):
                self._upload_to_s3(tarball_path, output_path)
                logger.info("Uploaded SageMaker artifact", s3_uri=output_path)
                return output_path
            else:
                # Copy to local output
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy2(tarball_path, str(out))
                logger.info("Created SageMaker artifact", path=output_path)
                return output_path

    def package_for_bedrock(
        self,
        model_path: str,
        output_s3_uri: str,
    ) -> str:
        """Upload model artifacts to S3 in Bedrock-expected structure.

        Validates model architecture, file presence, and size limits
        before uploading.

        Args:
            model_path: Local path to model directory.
            output_s3_uri: S3 URI for the Bedrock model.

        Returns:
            The S3 URI of the uploaded artifacts.

        Raises:
            ValueError: If model fails validation.
        """
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        # Validate before upload
        validation = self.verify_artifact(model_path, "bedrock")
        if not validation["is_valid"]:
            raise ValueError(
                f"Model artifacts invalid for Bedrock: {validation['issues']}"
            )

        # Upload files to S3
        if boto3 is None:
            raise ImportError("boto3 is required for S3 uploads")

        bucket, prefix = self._parse_s3_uri(output_s3_uri)
        s3_client: Any = boto3.client("s3")

        uploaded_files: list[str] = []
        for fpath in model_dir.iterdir():
            if fpath.is_file():
                key = f"{prefix}/{fpath.name}"
                s3_client.upload_file(str(fpath), bucket, key)
                uploaded_files.append(fpath.name)

        logger.info(
            "Uploaded Bedrock artifacts",
            s3_uri=output_s3_uri,
            file_count=len(uploaded_files),
        )
        return output_s3_uri

    def verify_artifact(self, path: str, target: str) -> dict[str, Any]:
        """Verify artifact integrity for given target (sagemaker or bedrock).

        Checks:
        - All required files present
        - safetensors file integrity (existence check)
        - Total size within limits

        Args:
            path: Path to artifact (directory or tar.gz).
            target: Deployment target ('sagemaker' or 'bedrock').

        Returns:
            Dict with is_valid, issues list, and size_gb.
        """
        issues: list[str] = []
        total_size_bytes = 0

        if target == "sagemaker":
            return self._verify_sagemaker_artifact(path)
        elif target == "bedrock":
            return self._verify_bedrock_artifact(path)
        else:
            return {
                "is_valid": False,
                "issues": [f"Unknown target: {target}"],
                "size_gb": 0.0,
            }

    # ── Private helpers ─────────────────────────────────────────

    def _verify_sagemaker_artifact(self, path: str) -> dict[str, Any]:
        """Verify SageMaker model.tar.gz artifact."""
        issues: list[str] = []
        total_size_bytes = 0

        artifact_path = Path(path)

        if artifact_path.is_file() and path.endswith(".tar.gz"):
            # Verify tar.gz contents
            try:
                with tarfile.open(path, "r:gz") as tar:
                    members = tar.getnames()
                    total_size_bytes = sum(m.size for m in tar.getmembers())

                    has_model = any(m.startswith("model/") for m in members)
                    has_code = any(m.startswith("code/") for m in members)

                    if not has_model:
                        issues.append("Missing model/ directory in archive")
                    if not has_code:
                        issues.append("Missing code/ directory in archive")
            except tarfile.TarError as e:
                issues.append(f"Invalid tar.gz file: {e}")
        elif artifact_path.is_dir():
            # Verify directory contents for pre-packaging
            config_path = artifact_path / "config.json"
            if not config_path.exists():
                issues.append("Missing config.json")

            safetensors = list(artifact_path.glob("*.safetensors"))
            if not safetensors:
                issues.append("No safetensors files found")

            for f in artifact_path.rglob("*"):
                if f.is_file():
                    total_size_bytes += f.stat().st_size
        else:
            issues.append(f"Path not found: {path}")

        size_gb = total_size_bytes / (1024**3)

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "size_gb": round(size_gb, 2),
        }

    def _verify_bedrock_artifact(self, path: str) -> dict[str, Any]:
        """Verify Bedrock model artifacts."""
        issues: list[str] = []
        total_size_bytes = 0

        model_dir = Path(path)
        if not model_dir.is_dir():
            return {
                "is_valid": False,
                "issues": [f"Directory not found: {path}"],
                "size_gb": 0.0,
            }

        # Check config.json
        config_path = model_dir / "config.json"
        if not config_path.exists():
            issues.append("Missing config.json")
        else:
            try:
                with open(config_path) as f:
                    config = json.load(f)
                architectures = config.get("architectures", [])
                if not architectures:
                    issues.append("config.json missing 'architectures' field")
                elif architectures[0] not in _SUPPORTED_ARCHITECTURES:
                    issues.append(
                        f"Unsupported architecture: {architectures[0]}. "
                        f"Supported: {_SUPPORTED_ARCHITECTURES}"
                    )
            except json.JSONDecodeError:
                issues.append("config.json is not valid JSON")

        # Check tokenizer files
        for tok_file in _TOKENIZER_FILES:
            if not (model_dir / tok_file).exists():
                issues.append(f"Missing tokenizer file: {tok_file}")

        # Check safetensors
        safetensors = list(model_dir.glob("*.safetensors"))
        if not safetensors:
            issues.append("No safetensors files found")

        # Calculate total size
        for f in model_dir.rglob("*"):
            if f.is_file():
                total_size_bytes += f.stat().st_size

        size_gb = total_size_bytes / (1024**3)
        if size_gb > _BEDROCK_MAX_SIZE_GB:
            issues.append(
                f"Total size {size_gb:.1f}GB exceeds Bedrock limit of "
                f"{_BEDROCK_MAX_SIZE_GB}GB"
            )

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "size_gb": round(size_gb, 2),
        }

    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and prefix.

        Args:
            s3_uri: S3 URI (s3://bucket/prefix).

        Returns:
            Tuple of (bucket, prefix).
        """
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    @staticmethod
    def _upload_to_s3(local_path: str, s3_uri: str) -> None:
        """Upload a local file to S3.

        Args:
            local_path: Path to local file.
            s3_uri: Destination S3 URI.
        """
        if boto3 is None:
            raise ImportError("boto3 is required for S3 uploads")

        bucket, key = ArtifactPackager._parse_s3_uri(s3_uri)
        s3_client: Any = boto3.client("s3")
        s3_client.upload_file(local_path, bucket, key)


__all__: list[str] = ["ArtifactPackager"]
