"""Unified model evaluation framework."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from lm_eval import evaluator as lm_evaluator
    from lm_eval.models.huggingface import HFLM

    _LM_EVAL_AVAILABLE = True
except ImportError:
    _LM_EVAL_AVAILABLE = False

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class ModelEvaluator:
    """Evaluate fine-tuned models with multiple strategies."""

    def __init__(self, model_path: str, device_map: str = "auto") -> None:
        """Load model (merged or adapter) for evaluation.

        - Support local paths and S3 URIs
        - If path contains adapter_config.json, load as PEFT adapter
        - Otherwise load as full model
        - Set model to eval mode
        - Log: model size, device, dtype

        Args:
            model_path: Local path or S3 URI to model artifacts.
            device_map: Device mapping strategy for model loading.
        """
        self.model_path = model_path
        self.device_map = device_map
        self.model: Any = None
        self.tokenizer: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer from the configured path."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_path = self._resolve_path(self.model_path)
        adapter_config = Path(local_path) / "adapter_config.json"

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if adapter_config.exists():
            # Load as PEFT adapter
            from peft import PeftModel, PeftConfig

            peft_config = PeftConfig.from_pretrained(local_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map=self.device_map,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base_model, local_path)
            logger.info("Loaded PEFT adapter model", path=local_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                device_map=self.device_map,
                trust_remote_code=True,
            )
            logger.info("Loaded full model", path=local_path)

        self.model.eval()

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            "Model loaded for evaluation",
            total_params=total_params,
            trainable_params=trainable_params,
            dtype=str(next(self.model.parameters()).dtype),
        )

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve S3 URI to local path or return as-is.

        Args:
            path: Local path or S3 URI.

        Returns:
            Local file path.
        """
        if path.startswith("s3://"):
            import tempfile

            import boto3

            local_dir = tempfile.mkdtemp(prefix="eval_model_")
            parts = path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            s3 = boto3.client("s3")
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(prefix) :].lstrip("/")
                    local_file = Path(local_dir) / rel
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, key, str(local_file))
            logger.info("Downloaded model from S3", s3_path=path, local_path=local_dir)
            return local_dir
        return path

    def evaluate_perplexity(
        self,
        dataset: Any,
        batch_size: int = 8,
        stride: int = 512,
    ) -> float:
        """Compute perplexity on eval dataset using sliding window.

        Handle long sequences with stride-based evaluation.

        Args:
            dataset: Evaluation dataset with a ``text`` column or similar.
            batch_size: Batch size for evaluation.
            stride: Stride for sliding window on long sequences.

        Returns:
            Perplexity score as a float.
        """
        import numpy as np

        self.model.eval()
        losses: list[float] = []

        texts = dataset["text"] if hasattr(dataset, "__getitem__") and "text" in dataset.column_names else [str(x) for x in dataset]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.tokenizer.model_max_length,
            )

            with torch.no_grad():
                input_ids = encodings["input_ids"].to(self.model.device)
                attention_mask = encodings["attention_mask"].to(self.model.device)
                seq_len = input_ids.size(1)

                # Sliding window for long sequences
                for begin_loc in range(0, seq_len, stride):
                    end_loc = min(begin_loc + self.tokenizer.model_max_length, seq_len)
                    input_chunk = input_ids[:, begin_loc:end_loc]
                    mask_chunk = attention_mask[:, begin_loc:end_loc]
                    labels = input_chunk.clone()

                    outputs = self.model(
                        input_ids=input_chunk,
                        attention_mask=mask_chunk,
                        labels=labels,
                    )
                    losses.append(outputs.loss.item())

                    if end_loc >= seq_len:
                        break

        avg_loss = float(np.mean(losses))
        perplexity = float(np.exp(avg_loss))
        logger.info("Perplexity evaluated", perplexity=perplexity, num_batches=len(losses))
        return perplexity

    def evaluate_generation(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> list[dict[str, Any]]:
        """Generate responses for a list of prompts.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            do_sample: Whether to use sampling.

        Returns:
            List of dicts with: prompt, generated_text, num_tokens,
            latency_ms, tokens_per_second.
        """
        results: list[dict[str, Any]] = []

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            start_time = time.perf_counter()
            with torch.no_grad():
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    gen_kwargs["top_p"] = top_p
                outputs = self.model.generate(**inputs, **gen_kwargs)
            elapsed = (time.perf_counter() - start_time) * 1000  # ms

            generated_ids = outputs[0][input_len:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            num_tokens = len(generated_ids)
            tokens_per_second = (num_tokens / (elapsed / 1000)) if elapsed > 0 else 0.0

            results.append(
                {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "num_tokens": num_tokens,
                    "latency_ms": round(elapsed, 2),
                    "tokens_per_second": round(tokens_per_second, 2),
                }
            )

        logger.info("Generation evaluation complete", num_prompts=len(prompts))
        return results

    def evaluate_benchmarks(
        self,
        benchmarks: list[str],
        num_fewshot: int = 5,
    ) -> dict[str, Any]:
        """Run lm-evaluation-harness benchmarks.

        Supported: mmlu, hellaswag, arc_easy, arc_challenge, winogrande,
        truthfulqa_mc, gsm8k.

        Falls back gracefully if lm-eval not installed.

        Args:
            benchmarks: List of benchmark names to run.
            num_fewshot: Number of few-shot examples.

        Returns:
            Results per benchmark with scores.
        """
        if not _LM_EVAL_AVAILABLE:
            logger.warning(
                "lm-eval not installed — skipping benchmark evaluation"
            )
            return {"error": "lm-eval not installed"}

        lm = HFLM(pretrained=self.model, tokenizer=self.tokenizer)

        results = lm_evaluator.simple_evaluate(
            model=lm,
            tasks=benchmarks,
            num_fewshot=num_fewshot,
        )

        benchmark_results: dict[str, Any] = {}
        for task_name, task_results in results.get("results", {}).items():
            benchmark_results[task_name] = {
                k: v for k, v in task_results.items() if isinstance(v, (int, float))
            }

        logger.info(
            "Benchmark evaluation complete",
            benchmarks=benchmarks,
            results=benchmark_results,
        )
        return benchmark_results

    def evaluate_custom_task(
        self,
        eval_dataset_path: str,
        metrics: list[str],
        max_new_tokens: int = 512,
    ) -> dict[str, float]:
        """Run custom evaluation task.

        Load dataset from path, generate responses, score against references.
        Metrics: exact_match, bleu, rouge_l, f1_token.

        Args:
            eval_dataset_path: Path to evaluation dataset (JSONL).
            metrics: List of metrics to compute.
            max_new_tokens: Maximum new tokens to generate.

        Returns:
            Dictionary of metric scores.
        """
        import json as json_mod

        from src.evaluation.metrics import MetricsComputer

        # Load dataset
        path = Path(eval_dataset_path)
        samples: list[dict[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json_mod.loads(line))

        prompts = [s.get("input", s.get("prompt", "")) for s in samples]
        references = [s.get("output", s.get("reference", "")) for s in samples]

        # Generate responses
        gen_results = self.evaluate_generation(
            prompts, max_new_tokens=max_new_tokens, do_sample=False
        )
        predictions = [r["generated_text"] for r in gen_results]

        # Compute requested metrics
        results: dict[str, float] = {}

        if "exact_match" in metrics:
            from src.evaluation.metrics import compute_exact_match

            results["exact_match"] = compute_exact_match(predictions, references)

        if "bleu" in metrics:
            bleu_results = MetricsComputer.compute_bleu(
                predictions, [[r] for r in references]
            )
            results["bleu"] = float(bleu_results.get("bleu", 0.0))

        if "rouge_l" in metrics:
            rouge_results = MetricsComputer.compute_rouge(predictions, references)
            results["rouge_l"] = float(rouge_results.get("rougeL", 0.0))

        if "f1_token" in metrics:
            from src.evaluation.metrics import compute_batch_f1

            results["f1_token"] = compute_batch_f1(predictions, references)

        logger.info(
            "Custom task evaluation complete",
            dataset=eval_dataset_path,
            metrics=results,
        )
        return results

    def run_full_evaluation(self, eval_config: dict[str, Any]) -> dict[str, Any]:
        """Run all configured evaluations.

        eval_config should match configs/evaluation/eval_default.yaml schema.
        Aggregate all results into single dict.
        Log everything to MLflow if available.

        Args:
            eval_config: Evaluation configuration dictionary.

        Returns:
            Complete evaluation results dictionary.
        """
        all_results: dict[str, Any] = {}

        # Perplexity
        ppl_config = eval_config.get("perplexity", {})
        if ppl_config:
            try:
                from datasets import load_dataset

                ds_split = ppl_config.get("dataset_split", "test")
                # Attempt to load a standard dataset for perplexity
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=ds_split)
                ppl = self.evaluate_perplexity(
                    ds,
                    batch_size=ppl_config.get("batch_size", 8),
                    stride=ppl_config.get("stride", 512),
                )
                all_results["perplexity"] = ppl
            except Exception:
                logger.warning("Perplexity evaluation failed", exc_info=True)

        # Benchmarks
        benchmarks = eval_config.get("benchmarks", [])
        if benchmarks:
            benchmark_results = self.evaluate_benchmarks(benchmarks)
            all_results["benchmarks"] = benchmark_results

        # Generation eval
        gen_config = eval_config.get("generation_eval", {})
        prompts_path = gen_config.get("prompts_path", "")
        if prompts_path:
            import json as json_mod

            with open(prompts_path, encoding="utf-8") as f:
                prompts = [json_mod.loads(line)["prompt"] for line in f if line.strip()]
            gen_results = self.evaluate_generation(
                prompts,
                max_new_tokens=gen_config.get("max_new_tokens", 512),
                temperature=gen_config.get("temperature", 0.1),
                top_p=gen_config.get("top_p", 0.9),
                do_sample=gen_config.get("do_sample", False),
            )
            all_results["generation"] = gen_results

        # Custom tasks
        custom_tasks = eval_config.get("custom_tasks", [])
        for task in custom_tasks:
            task_name = task.get("name", "custom")
            task_results = self.evaluate_custom_task(
                eval_dataset_path=task["dataset_path"],
                metrics=task.get("metrics", ["exact_match"]),
                max_new_tokens=task.get("max_new_tokens", 512),
            )
            all_results[f"custom_{task_name}"] = task_results

        # Log to MLflow if available
        if _MLFLOW_AVAILABLE:
            try:
                flat_metrics: dict[str, float] = {}
                if "perplexity" in all_results:
                    flat_metrics["eval_perplexity"] = all_results["perplexity"]
                for bm_name, bm_scores in all_results.get("benchmarks", {}).items():
                    if isinstance(bm_scores, dict):
                        for k, v in bm_scores.items():
                            if isinstance(v, (int, float)):
                                flat_metrics[f"benchmark_{bm_name}_{k}"] = float(v)
                if flat_metrics:
                    mlflow.log_metrics(flat_metrics)
            except Exception:
                logger.warning("Failed to log eval results to MLflow", exc_info=True)

        logger.info("Full evaluation complete", result_keys=list(all_results.keys()))
        return all_results


__all__: list[str] = ["ModelEvaluator"]
