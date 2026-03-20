"""Merge LoRA/DoRA adapters into base models for deployment."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import boto3
import structlog
import torch
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = structlog.get_logger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ═══════════════════════════════════════════════════════════════
# MergeResult
# ═══════════════════════════════════════════════════════════════


class MergeResult(BaseModel):
    """Immutable record of a completed adapter merge."""

    merged_model_path: str
    model_size_gb: float
    num_parameters: int
    verification_output: str
    safetensors_files: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# AdapterMerger
# ═══════════════════════════════════════════════════════════════


class AdapterMerger:
    """Merge LoRA/DoRA adapters into a base model and export as safetensors."""

    def merge_adapter(
        self,
        base_model_name: str,
        adapter_path: str,
        output_path: str,
        dtype: str = "float16",
        device_map: str = "auto",
        push_to_hub: bool = False,
        hub_model_id: str | None = None,
    ) -> MergeResult:
        """Merge adapter weights into the base model.

        Args:
            base_model_name: HuggingFace model ID or local path to base model.
            adapter_path: Local path or ``s3://`` URI to adapter weights.
            output_path: Local path or ``s3://`` URI for merged model output.
            dtype: Target dtype (``float16`` or ``bfloat16``).
            device_map: Device placement strategy.
            push_to_hub: Whether to push the merged model to HuggingFace Hub.
            hub_model_id: HuggingFace Hub repo ID (required when *push_to_hub*).

        Returns:
            :class:`MergeResult` with paths, sizes, and verification output.
        """
        # 1. Download adapter from S3 if needed
        local_adapter = self._resolve_s3_input(adapter_path)

        # Resolve output — use a tempdir if output is S3
        is_s3_output = output_path.startswith("s3://")
        local_output = (
            tempfile.mkdtemp(prefix="merged-model-") if is_s3_output else output_path
        )

        torch_dtype = _DTYPE_MAP[dtype]

        # 2. Load base model in full precision (no quantization)
        logger.info(
            "Loading base model for merging",
            model=base_model_name,
            dtype=dtype,
            device_map=device_map,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

        # 3. Load adapter
        logger.info("Loading adapter", path=local_adapter)
        model = PeftModel.from_pretrained(model, local_adapter)

        # 4. Merge adapter into base model
        logger.info("Merging adapter weights")
        model = model.merge_and_unload()

        # 5. Save as safetensors
        logger.info("Saving merged model", output=local_output)
        Path(local_output).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_output, safe_serialization=True)

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(local_output)

        # 6. Verify merged model
        verification_output = self._verify_model(local_output, torch_dtype, device_map)

        # 7. Calculate model size and parameter count
        num_parameters = sum(p.numel() for p in model.parameters())
        model_size_gb = self._calculate_model_size(local_output)
        safetensors_files = self._list_safetensors(local_output)

        logger.info(
            "Merge complete",
            num_parameters=num_parameters,
            model_size_gb=round(model_size_gb, 2),
            safetensors_count=len(safetensors_files),
        )

        # 8. Push to hub
        if push_to_hub:
            if hub_model_id is None:
                raise ValueError("hub_model_id is required when push_to_hub is True")
            logger.info("Pushing to HuggingFace Hub", repo=hub_model_id)
            model.push_to_hub(hub_model_id, safe_serialization=True)
            tokenizer.push_to_hub(hub_model_id)

        # 9. Upload to S3 if requested
        final_path = output_path
        if is_s3_output:
            self._upload_to_s3(local_output, output_path)
            final_path = output_path

        return MergeResult(
            merged_model_path=final_path,
            model_size_gb=round(model_size_gb, 4),
            num_parameters=num_parameters,
            verification_output=verification_output,
            safetensors_files=safetensors_files,
        )

    # ── private helpers ─────────────────────────────────────────

    @staticmethod
    def _resolve_s3_input(path: str) -> str:
        """Download from S3 to a local temp dir if *path* is an S3 URI."""
        if not path.startswith("s3://"):
            return path

        parts = path.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        local_dir = tempfile.mkdtemp(prefix="adapter-download-")
        s3 = boto3.client("s3")

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):].lstrip("/")
                if not rel:
                    continue
                dest = Path(local_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dest))

        logger.info("Downloaded adapter from S3", s3_uri=path, local_dir=local_dir)
        return local_dir

    @staticmethod
    def _verify_model(
        model_path: str,
        torch_dtype: torch.dtype,
        device_map: str,
    ) -> str:
        """Reload the merged model and run a short test generation."""
        logger.info("Verifying merged model", path=model_path)
        verify_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        verify_tokenizer = AutoTokenizer.from_pretrained(model_path)

        prompt = "The capital of France is"
        inputs = verify_tokenizer(prompt, return_tensors="pt")
        # Move inputs to model device
        device = next(verify_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = verify_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )
        generated = verify_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Free verification model memory
        del verify_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Verification output", text=generated)
        return generated

    @staticmethod
    def _calculate_model_size(model_dir: str) -> float:
        """Sum the size of all files in *model_dir* and return in GB."""
        total_bytes = 0
        for f in Path(model_dir).rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
        return total_bytes / (1024 ** 3)

    @staticmethod
    def _list_safetensors(model_dir: str) -> list[str]:
        """Return a sorted list of safetensors filenames."""
        return sorted(
            f.name for f in Path(model_dir).glob("*.safetensors")
        )

    @staticmethod
    def _upload_to_s3(local_dir: str, s3_uri: str) -> None:
        """Upload all files from *local_dir* to *s3_uri*."""
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        s3 = boto3.client("s3")
        for file_path in Path(local_dir).rglob("*"):
            if file_path.is_file():
                key = f"{prefix}/{file_path.relative_to(local_dir)}"
                s3.upload_file(str(file_path), bucket, key)

        logger.info("Uploaded merged model to S3", s3_uri=s3_uri)
