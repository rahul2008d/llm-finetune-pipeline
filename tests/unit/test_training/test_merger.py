"""Unit tests for training.merger – AdapterMerger and MergeResult."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError

from src.training.merger import AdapterMerger, MergeResult, _DTYPE_MAP


# ── helpers ─────────────────────────────────────────────────────


def _fake_model(num_params: int = 1000) -> MagicMock:
    """Build a mock model with controllable parameter count."""
    param = MagicMock()
    param.numel.return_value = num_params
    param.device = torch.device("cpu")

    model = MagicMock()
    model.parameters.return_value = [param]
    model.merge_and_unload.return_value = model
    return model


def _fake_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tok.decode.return_value = "The capital of France is Paris"
    return tok


@pytest.fixture()
def merger() -> AdapterMerger:
    return AdapterMerger()


# ═══════════════════════════════════════════════════════════════
# MergeResult
# ═══════════════════════════════════════════════════════════════


class TestMergeResult:
    def test_valid_construction(self) -> None:
        r = MergeResult(
            merged_model_path="/tmp/merged",
            model_size_gb=13.5,
            num_parameters=7_000_000_000,
            verification_output="The capital of France is Paris",
            safetensors_files=["model-00001.safetensors", "model-00002.safetensors"],
        )
        assert r.merged_model_path == "/tmp/merged"
        assert r.model_size_gb == pytest.approx(13.5)
        assert r.num_parameters == 7_000_000_000
        assert len(r.safetensors_files) == 2

    def test_default_empty_safetensors(self) -> None:
        r = MergeResult(
            merged_model_path="/tmp/m",
            model_size_gb=0.0,
            num_parameters=0,
            verification_output="",
        )
        assert r.safetensors_files == []

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            MergeResult(
                merged_model_path="/tmp/m",
                model_size_gb=1.0,
                # missing num_parameters and verification_output
            )

    def test_model_dump(self) -> None:
        r = MergeResult(
            merged_model_path="/out",
            model_size_gb=6.5,
            num_parameters=3_000_000,
            verification_output="ok",
            safetensors_files=["model.safetensors"],
        )
        d = r.model_dump()
        assert d["merged_model_path"] == "/out"
        assert d["safetensors_files"] == ["model.safetensors"]


# ═══════════════════════════════════════════════════════════════
# _DTYPE_MAP
# ═══════════════════════════════════════════════════════════════


class TestDtypeMap:
    def test_float16(self) -> None:
        assert _DTYPE_MAP["float16"] is torch.float16

    def test_bfloat16(self) -> None:
        assert _DTYPE_MAP["bfloat16"] is torch.bfloat16

    def test_float32(self) -> None:
        assert _DTYPE_MAP["float32"] is torch.float32


# ═══════════════════════════════════════════════════════════════
# _resolve_s3_input
# ═══════════════════════════════════════════════════════════════


class TestResolveS3Input:
    def test_local_path_returned_unchanged(self) -> None:
        result = AdapterMerger._resolve_s3_input("/local/adapter")
        assert result == "/local/adapter"

    @patch("training.merger.boto3")
    def test_s3_path_downloads_files(self, mock_boto: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_boto.client.return_value = mock_s3

        paginator = MagicMock()
        mock_s3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "adapters/v1/adapter_model.bin"},
                    {"Key": "adapters/v1/config.json"},
                ],
            },
        ]

        result = AdapterMerger._resolve_s3_input("s3://bucket/adapters/v1")

        assert not result.startswith("s3://")
        assert mock_s3.download_file.call_count == 2

    @patch("training.merger.boto3")
    def test_s3_empty_contents(self, mock_boto: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_boto.client.return_value = mock_s3

        paginator = MagicMock()
        mock_s3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"Contents": []}]

        result = AdapterMerger._resolve_s3_input("s3://bucket/empty")
        assert not result.startswith("s3://")


# ═══════════════════════════════════════════════════════════════
# _calculate_model_size
# ═══════════════════════════════════════════════════════════════


class TestCalculateModelSize:
    def test_sums_file_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with known sizes
            f1 = Path(tmpdir) / "model.safetensors"
            f1.write_bytes(b"\x00" * 1024)  # 1 KB
            f2 = Path(tmpdir) / "config.json"
            f2.write_bytes(b"\x00" * 512)  # 0.5 KB

            size_gb = AdapterMerger._calculate_model_size(tmpdir)

            expected = (1024 + 512) / (1024 ** 3)
            assert size_gb == pytest.approx(expected, abs=1e-10)

    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            assert AdapterMerger._calculate_model_size(tmpdir) == 0.0

    def test_includes_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "sub"
            subdir.mkdir()
            (subdir / "weights.bin").write_bytes(b"\x00" * 2048)

            size_gb = AdapterMerger._calculate_model_size(tmpdir)
            assert size_gb > 0


# ═══════════════════════════════════════════════════════════════
# _list_safetensors
# ═══════════════════════════════════════════════════════════════


class TestListSafetensors:
    def test_lists_safetensors_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model-00001.safetensors").touch()
            (Path(tmpdir) / "model-00002.safetensors").touch()
            (Path(tmpdir) / "config.json").touch()

            result = AdapterMerger._list_safetensors(tmpdir)

            assert result == ["model-00001.safetensors", "model-00002.safetensors"]

    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            assert AdapterMerger._list_safetensors(tmpdir) == []

    def test_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "z.safetensors").touch()
            (Path(tmpdir) / "a.safetensors").touch()

            result = AdapterMerger._list_safetensors(tmpdir)
            assert result == ["a.safetensors", "z.safetensors"]


# ═══════════════════════════════════════════════════════════════
# _upload_to_s3
# ═══════════════════════════════════════════════════════════════


class TestUploadToS3:
    @patch("training.merger.boto3")
    def test_uploads_all_files(self, mock_boto: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_boto.client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.safetensors").write_text("data")
            (Path(tmpdir) / "config.json").write_text("{}")

            AdapterMerger._upload_to_s3(tmpdir, "s3://bucket/models/merged")

        assert mock_s3.upload_file.call_count == 2

    @patch("training.merger.boto3")
    def test_correct_bucket_and_prefix(self, mock_boto: MagicMock) -> None:
        mock_s3 = MagicMock()
        mock_boto.client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model.safetensors").write_text("data")

            AdapterMerger._upload_to_s3(tmpdir, "s3://my-bucket/prefix/v1")

        call_args = mock_s3.upload_file.call_args
        # Second arg is bucket
        assert call_args[0][1] == "my-bucket"
        # Third arg (key) starts with prefix
        assert call_args[0][2].startswith("prefix/v1/")


# ═══════════════════════════════════════════════════════════════
# _verify_model
# ═══════════════════════════════════════════════════════════════


class TestVerifyModel:
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.AutoModelForCausalLM")
    @patch("training.merger.torch")
    def test_returns_generated_text(
        self,
        mock_torch: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_tok: MagicMock,
    ) -> None:
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        model = MagicMock()
        param = MagicMock()
        param.device = torch.device("cpu")
        model.parameters.return_value = iter([param])
        model.generate.return_value = [torch.tensor([1, 2, 3])]
        mock_auto_model.from_pretrained.return_value = model

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        tokenizer.decode.return_value = "The capital of France is Paris"
        mock_auto_tok.from_pretrained.return_value = tokenizer

        result = AdapterMerger._verify_model(
            "/tmp/model", torch.float16, "auto",
        )

        assert "Paris" in result
        mock_auto_model.from_pretrained.assert_called_once()
        model.generate.assert_called_once()

    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.AutoModelForCausalLM")
    @patch("training.merger.torch")
    def test_moves_inputs_to_device(
        self,
        mock_torch: MagicMock,
        mock_auto_model: MagicMock,
        mock_auto_tok: MagicMock,
    ) -> None:
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        model = MagicMock()
        param = MagicMock()
        param.device = torch.device("cpu")
        model.parameters.return_value = iter([param])
        model.generate.return_value = [torch.tensor([1])]
        mock_auto_model.from_pretrained.return_value = model

        # Tokenizer returns tensors with .to method
        input_tensor = MagicMock()
        input_tensor.to.return_value = input_tensor
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": input_tensor}
        tokenizer.decode.return_value = "output"
        mock_auto_tok.from_pretrained.return_value = tokenizer

        AdapterMerger._verify_model("/tmp/m", torch.float16, "auto")

        input_tensor.to.assert_called_once()


# ═══════════════════════════════════════════════════════════════
# merge_adapter (full pipeline)
# ═══════════════════════════════════════════════════════════════


class TestMergeAdapter:
    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_local_merge_happy_path(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            # Setup mocks
            base_model = _fake_model(num_params=5000)
            mock_auto_model.from_pretrained.return_value = base_model

            merged_model = _fake_model(num_params=5000)
            # When save_pretrained is called, create a safetensors file
            def save_side_effect(path: str, **kwargs: Any) -> None:
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "model.safetensors").write_bytes(b"\x00" * 1024)

            merged_model.save_pretrained.side_effect = save_side_effect
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = merged_model
            peft_model.parameters.return_value = merged_model.parameters()
            mock_peft.from_pretrained.return_value = peft_model

            mock_auto_tok.from_pretrained.return_value = MagicMock()
            mock_verify.return_value = "The capital of France is Paris"

            result = merger.merge_adapter(
                base_model_name="meta-llama/Llama-3.1-8B-Instruct",
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            assert isinstance(result, MergeResult)
            assert result.merged_model_path == str(output_dir)
            assert result.verification_output == "The capital of France is Paris"
            assert "model.safetensors" in result.safetensors_files
            assert result.num_parameters == 5000
            assert result.model_size_gb >= 0

            # Verify calls
            mock_auto_model.from_pretrained.assert_called_once_with(
                "meta-llama/Llama-3.1-8B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            mock_peft.from_pretrained.assert_called_once()
            peft_model.merge_and_unload.assert_called_once()
            merged_model.save_pretrained.assert_called_once_with(
                str(output_dir), safe_serialization=True,
            )

    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_bfloat16_dtype(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            model = _fake_model()
            model.save_pretrained.side_effect = lambda p, **kw: (
                Path(p).mkdir(parents=True, exist_ok=True)
            )
            mock_auto_model.from_pretrained.return_value = model
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = model
            peft_model.parameters.return_value = model.parameters()
            mock_peft.from_pretrained.return_value = peft_model
            mock_auto_tok.from_pretrained.return_value = MagicMock()
            mock_verify.return_value = "ok"

            merger.merge_adapter(
                base_model_name="model",
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
                dtype="bfloat16",
            )

            call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
            assert call_kwargs["torch_dtype"] is torch.bfloat16

    @patch("training.merger.AdapterMerger._upload_to_s3")
    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_s3_output_uploads(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        mock_upload: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            model = _fake_model()
            model.save_pretrained.side_effect = lambda p, **kw: (
                Path(p).mkdir(parents=True, exist_ok=True)
            )
            mock_auto_model.from_pretrained.return_value = model
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = model
            peft_model.parameters.return_value = model.parameters()
            mock_peft.from_pretrained.return_value = peft_model
            mock_auto_tok.from_pretrained.return_value = MagicMock()
            mock_verify.return_value = "verified"

            result = merger.merge_adapter(
                base_model_name="model",
                adapter_path=str(adapter_dir),
                output_path="s3://bucket/models/merged-v1",
            )

            mock_upload.assert_called_once()
            assert result.merged_model_path == "s3://bucket/models/merged-v1"

    @patch("training.merger.AdapterMerger._resolve_s3_input")
    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_s3_adapter_input_resolved(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        mock_resolve: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_adapter = Path(tmpdir) / "adapter"
            local_adapter.mkdir()
            output_dir = Path(tmpdir) / "output"

            mock_resolve.return_value = str(local_adapter)

            model = _fake_model()
            model.save_pretrained.side_effect = lambda p, **kw: (
                Path(p).mkdir(parents=True, exist_ok=True)
            )
            mock_auto_model.from_pretrained.return_value = model
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = model
            peft_model.parameters.return_value = model.parameters()
            mock_peft.from_pretrained.return_value = peft_model
            mock_auto_tok.from_pretrained.return_value = MagicMock()
            mock_verify.return_value = "verified"

            merger.merge_adapter(
                base_model_name="model",
                adapter_path="s3://bucket/adapters/v1",
                output_path=str(output_dir),
            )

            mock_resolve.assert_called_once_with("s3://bucket/adapters/v1")
            # PeftModel loaded from the resolved local path
            peft_call = mock_peft.from_pretrained.call_args
            assert str(local_adapter) in str(peft_call)

    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_push_to_hub(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            model = _fake_model()
            model.save_pretrained.side_effect = lambda p, **kw: (
                Path(p).mkdir(parents=True, exist_ok=True)
            )
            mock_auto_model.from_pretrained.return_value = model
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = model
            peft_model.parameters.return_value = model.parameters()
            mock_peft.from_pretrained.return_value = peft_model
            tokenizer = MagicMock()
            mock_auto_tok.from_pretrained.return_value = tokenizer
            mock_verify.return_value = "ok"

            merger.merge_adapter(
                base_model_name="model",
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
                push_to_hub=True,
                hub_model_id="org/my-merged-model",
            )

            model.push_to_hub.assert_called_once_with(
                "org/my-merged-model", safe_serialization=True,
            )
            tokenizer.push_to_hub.assert_called_once_with("org/my-merged-model")

    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_push_to_hub_requires_model_id(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            model = _fake_model()
            model.save_pretrained.side_effect = lambda p, **kw: (
                Path(p).mkdir(parents=True, exist_ok=True)
            )
            mock_auto_model.from_pretrained.return_value = model
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = model
            peft_model.parameters.return_value = model.parameters()
            mock_peft.from_pretrained.return_value = peft_model
            mock_auto_tok.from_pretrained.return_value = MagicMock()
            mock_verify.return_value = "ok"

            with pytest.raises(ValueError, match="hub_model_id is required"):
                merger.merge_adapter(
                    base_model_name="model",
                    adapter_path=str(adapter_dir),
                    output_path=str(output_dir),
                    push_to_hub=True,
                    hub_model_id=None,
                )

    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_custom_device_map(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            model = _fake_model()
            model.save_pretrained.side_effect = lambda p, **kw: (
                Path(p).mkdir(parents=True, exist_ok=True)
            )
            mock_auto_model.from_pretrained.return_value = model
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = model
            peft_model.parameters.return_value = model.parameters()
            mock_peft.from_pretrained.return_value = peft_model
            mock_auto_tok.from_pretrained.return_value = MagicMock()
            mock_verify.return_value = "ok"

            merger.merge_adapter(
                base_model_name="model",
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
                device_map="cpu",
            )

            call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
            assert call_kwargs["device_map"] == "cpu"

    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_safe_serialization_flag(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            model = _fake_model()
            model.save_pretrained.side_effect = lambda p, **kw: (
                Path(p).mkdir(parents=True, exist_ok=True)
            )
            mock_auto_model.from_pretrained.return_value = model
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = model
            peft_model.parameters.return_value = model.parameters()
            mock_peft.from_pretrained.return_value = peft_model
            mock_auto_tok.from_pretrained.return_value = MagicMock()
            mock_verify.return_value = "ok"

            merger.merge_adapter(
                base_model_name="model",
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            save_kwargs = model.save_pretrained.call_args
            assert save_kwargs.kwargs.get("safe_serialization") is True

    @patch("training.merger.AdapterMerger._verify_model")
    @patch("training.merger.AutoTokenizer")
    @patch("training.merger.PeftModel")
    @patch("training.merger.AutoModelForCausalLM")
    def test_tokenizer_saved(
        self,
        mock_auto_model: MagicMock,
        mock_peft: MagicMock,
        mock_auto_tok: MagicMock,
        mock_verify: MagicMock,
        merger: AdapterMerger,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            output_dir = Path(tmpdir) / "output"

            model = _fake_model()
            model.save_pretrained.side_effect = lambda p, **kw: (
                Path(p).mkdir(parents=True, exist_ok=True)
            )
            mock_auto_model.from_pretrained.return_value = model
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = model
            peft_model.parameters.return_value = model.parameters()
            mock_peft.from_pretrained.return_value = peft_model
            tokenizer = MagicMock()
            mock_auto_tok.from_pretrained.return_value = tokenizer
            mock_verify.return_value = "ok"

            merger.merge_adapter(
                base_model_name="base-model",
                adapter_path=str(adapter_dir),
                output_path=str(output_dir),
            )

            # Tokenizer loaded from base model
            mock_auto_tok.from_pretrained.assert_called_once_with("base-model")
            # Tokenizer saved to output
            tokenizer.save_pretrained.assert_called_once_with(str(output_dir))
