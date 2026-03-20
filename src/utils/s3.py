"""S3 client with retry logic, KMS encryption support, and multipart uploads.

Wraps boto3 S3 operations with tenacity-based retries and structured logging.
Files larger than 100 MB are uploaded via multipart transfer automatically.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import boto3
import botocore.exceptions
import structlog
from boto3.s3.transfer import TransferConfig
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

_MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100 MB
_MULTIPART_CHUNKSIZE = 100 * 1024 * 1024  # 100 MB


def _log_s3_retry(retry_state: RetryCallState) -> None:
    """Log S3 retry attempts with structured context.

    Args:
        retry_state: Tenacity retry state.
    """
    logger.warning(
        "Retrying S3 operation",
        attempt=retry_state.attempt_number,
        fn=getattr(retry_state.fn, "__name__", str(retry_state.fn)),
        exception=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


_s3_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=30),
    retry=retry_if_exception_type(
        (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError)
    ),
    before_sleep=_log_s3_retry,
    reraise=True,
)


class S3Client:
    """S3 helper wrapping boto3 with automatic retries and optional KMS encryption."""

    def __init__(
        self,
        session: boto3.Session | None = None,
        region: str | None = None,
    ) -> None:
        """Initialise the S3 client.

        Args:
            session: Optional pre-configured boto3 session.
            region: AWS region override.
        """
        sess = session or boto3.Session()
        self._client: Any = sess.client("s3", region_name=region)
        self._transfer_config = TransferConfig(
            multipart_threshold=_MULTIPART_THRESHOLD,
            multipart_chunksize=_MULTIPART_CHUNKSIZE,
        )

    def _extra_args(self, kms_key_id: str | None) -> dict[str, str]:
        """Build ExtraArgs dict for server-side encryption.

        Args:
            kms_key_id: Optional KMS key ID for SSE-KMS.

        Returns:
            ExtraArgs dictionary (may be empty).
        """
        if kms_key_id:
            return {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": kms_key_id,
            }
        return {}

    @_s3_retry
    def upload_file(
        self,
        local_path: str,
        bucket: str,
        key: str,
        kms_key_id: str | None = None,
    ) -> None:
        """Upload a file to S3 (multipart for files > 100 MB).

        Args:
            local_path: Path to the local file.
            bucket: Destination S3 bucket.
            key: Destination S3 object key.
            kms_key_id: Optional KMS key ID for encryption.
        """
        extra = self._extra_args(kms_key_id)
        logger.info("Uploading file", local_path=local_path, bucket=bucket, key=key)
        self._client.upload_file(
            local_path,
            bucket,
            key,
            Config=self._transfer_config,
            ExtraArgs=extra or None,
        )
        logger.info("Upload complete", key=key)

    @_s3_retry
    def download_file(
        self,
        bucket: str,
        key: str,
        local_path: str,
        kms_key_id: str | None = None,
    ) -> None:
        """Download a file from S3.

        Args:
            bucket: Source S3 bucket.
            key: Source S3 object key.
            local_path: Local path to write the file.
            kms_key_id: Unused for downloads; accepted for API symmetry.
        """
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading file", bucket=bucket, key=key, local_path=local_path)
        self._client.download_file(bucket, key, local_path)
        logger.info("Download complete", key=key)

    @_s3_retry
    def upload_directory(
        self,
        local_dir: str,
        bucket: str,
        prefix: str,
        kms_key_id: str | None = None,
    ) -> int:
        """Upload every file in a directory tree to S3.

        Args:
            local_dir: Root of the local directory.
            bucket: Destination S3 bucket.
            prefix: S3 key prefix.
            kms_key_id: Optional KMS key ID for encryption.

        Returns:
            Number of files uploaded.
        """
        extra = self._extra_args(kms_key_id)
        root = Path(local_dir)
        count = 0
        for file_path in root.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(root)
                s3_key = f"{prefix}/{relative}"
                self._client.upload_file(
                    str(file_path),
                    bucket,
                    s3_key,
                    Config=self._transfer_config,
                    ExtraArgs=extra or None,
                )
                count += 1
        logger.info("Directory uploaded", local_dir=local_dir, files=count)
        return count

    @_s3_retry
    def download_directory(
        self,
        bucket: str,
        prefix: str,
        local_dir: str,
        kms_key_id: str | None = None,
    ) -> int:
        """Download all objects under a prefix to a local directory.

        Args:
            bucket: Source S3 bucket.
            prefix: S3 key prefix.
            local_dir: Local directory to write files into.
            kms_key_id: Unused for downloads; accepted for API symmetry.

        Returns:
            Number of files downloaded.
        """
        keys = self.list_objects(bucket, prefix)
        root = Path(local_dir)
        count = 0
        for key in keys:
            relative = key[len(prefix) :].lstrip("/")
            if not relative:
                continue
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(bucket, key, str(dest))
            count += 1
        logger.info("Directory downloaded", prefix=prefix, files=count)
        return count

    @_s3_retry
    def list_objects(
        self,
        bucket: str,
        prefix: str,
        kms_key_id: str | None = None,
    ) -> list[str]:
        """List object keys under a prefix (handles pagination).

        Args:
            bucket: S3 bucket.
            prefix: Key prefix to filter by.
            kms_key_id: Unused; accepted for API symmetry.

        Returns:
            List of matching S3 keys.
        """
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        logger.info("Listed objects", bucket=bucket, prefix=prefix, count=len(keys))
        return keys

    @_s3_retry
    def check_exists(
        self,
        bucket: str,
        key: str,
        kms_key_id: str | None = None,
    ) -> bool:
        """Check whether an object exists in S3.

        Args:
            bucket: S3 bucket.
            key: S3 object key.
            kms_key_id: Unused; accepted for API symmetry.

        Returns:
            ``True`` if the object exists.
        """
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    @_s3_retry
    def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        expiration: int = 3600,
        kms_key_id: str | None = None,
    ) -> str:
        """Generate a presigned URL for an S3 object.

        Args:
            bucket: S3 bucket.
            key: S3 object key.
            expiration: URL lifetime in seconds (default 1 hour).
            kms_key_id: Unused; accepted for API symmetry.

        Returns:
            Presigned URL string.
        """
        url: str = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration,
        )
        logger.info("Presigned URL generated", bucket=bucket, key=key, expiration=expiration)
        return url


# Backward-compatible alias used in training.callbacks
S3Helper = S3Client

__all__: list[str] = ["S3Client", "S3Helper"]
