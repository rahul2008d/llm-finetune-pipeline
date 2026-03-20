"""Unit tests for utils.s3 module using moto for AWS mocking."""

from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from src.utils.s3 import S3Client


@pytest.fixture()
def _s3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set minimal AWS env vars for moto."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture()
def s3_bucket(_s3_env: None) -> str:
    """Create a mocked S3 bucket and return its name."""
    bucket_name = "test-bucket"
    with mock_aws():
        conn = boto3.client("s3", region_name="us-east-1")
        conn.create_bucket(Bucket=bucket_name)
        yield bucket_name  # type: ignore[misc]


class TestS3Client:
    """Tests for S3Client with moto."""

    @mock_aws
    def test_upload_and_download_file(self, _s3_env: None, tmp_path: Path) -> None:
        """Test round-trip upload then download."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")

        local_file = tmp_path / "hello.txt"
        local_file.write_text("hello world")

        session = boto3.Session(region_name="us-east-1")
        client = S3Client(session=session)
        client.upload_file(str(local_file), "test-bucket", "files/hello.txt")

        dest = tmp_path / "downloaded.txt"
        client.download_file("test-bucket", "files/hello.txt", str(dest))
        assert dest.read_text() == "hello world"

    @mock_aws
    def test_upload_directory(self, _s3_env: None, tmp_path: Path) -> None:
        """Test uploading an entire directory."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")

        (tmp_path / "a.txt").write_text("a")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("b")

        session = boto3.Session(region_name="us-east-1")
        client = S3Client(session=session)
        count = client.upload_directory(str(tmp_path), "test-bucket", "prefix")
        assert count == 2

    @mock_aws
    def test_download_directory(self, _s3_env: None, tmp_path: Path) -> None:
        """Test downloading all objects under a prefix."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="dir/one.txt", Body=b"1")
        s3.put_object(Bucket="test-bucket", Key="dir/two.txt", Body=b"2")

        session = boto3.Session(region_name="us-east-1")
        client = S3Client(session=session)
        count = client.download_directory("test-bucket", "dir/", str(tmp_path / "out"))
        assert count == 2
        assert (tmp_path / "out" / "one.txt").read_text() == "1"

    @mock_aws
    def test_list_objects(self, _s3_env: None) -> None:
        """Test listing objects with prefix."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="data/a.csv", Body=b"a")
        s3.put_object(Bucket="test-bucket", Key="data/b.csv", Body=b"b")

        session = boto3.Session(region_name="us-east-1")
        client = S3Client(session=session)
        keys = client.list_objects("test-bucket", "data/")
        assert sorted(keys) == ["data/a.csv", "data/b.csv"]

    @mock_aws
    def test_check_exists_true(self, _s3_env: None) -> None:
        """Test check_exists returns True for existing object."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="exists.txt", Body=b"yes")

        session = boto3.Session(region_name="us-east-1")
        client = S3Client(session=session)
        assert client.check_exists("test-bucket", "exists.txt") is True

    @mock_aws
    def test_check_exists_false(self, _s3_env: None) -> None:
        """Test check_exists returns False for missing object."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")

        session = boto3.Session(region_name="us-east-1")
        client = S3Client(session=session)
        assert client.check_exists("test-bucket", "nope.txt") is False

    @mock_aws
    def test_generate_presigned_url(self, _s3_env: None) -> None:
        """Test presigned URL generation."""
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="file.zip", Body=b"data")

        session = boto3.Session(region_name="us-east-1")
        client = S3Client(session=session)
        url = client.generate_presigned_url("test-bucket", "file.zip", expiration=60)
        assert "test-bucket" in url
        assert "file.zip" in url

    @mock_aws
    def test_upload_with_kms_key(self, _s3_env: None, tmp_path: Path) -> None:
        """Test upload_file passes KMS key in ExtraArgs."""
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")

        local_file = tmp_path / "secret.txt"
        local_file.write_text("encrypted")

        session = boto3.Session(region_name="us-east-1")
        client = S3Client(session=session)
        # moto accepts KMS args without error
        client.upload_file(
            str(local_file), "test-bucket", "secret.txt", kms_key_id="alias/my-key"
        )
