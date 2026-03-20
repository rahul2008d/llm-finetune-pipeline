"""Unit tests for evaluation.evaluator module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import pytest


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    def test_evaluate_perplexity_returns_float(self) -> None:
        """Mock model, verify positive float returned."""
        import torch
        from src.evaluation.evaluator import ModelEvaluator

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.model_max_length = 2048
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_output = MagicMock()
        mock_output.loss = MagicMock()
        mock_output.loss.item.return_value = 2.0
        mock_model.return_value = mock_output

        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer

        # Create mock dataset
        dataset = MagicMock()
        dataset.__getitem__ = MagicMock(return_value=["text1", "text2"])
        dataset.column_names = ["text"]
        dataset.__len__ = MagicMock(return_value=2)

        perplexity = evaluator.evaluate_perplexity(dataset, batch_size=2)
        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_evaluate_generation_returns_expected_format(self) -> None:
        """Verify dict keys in output."""
        import torch
        from src.evaluation.evaluator import ModelEvaluator

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer.decode.return_value = "Generated text"

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = [torch.ones(10, dtype=torch.long)]

        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer

        results = evaluator.evaluate_generation(["Hello"], max_new_tokens=50)

        assert len(results) == 1
        assert "prompt" in results[0]
        assert "generated_text" in results[0]
        assert "num_tokens" in results[0]
        assert "latency_ms" in results[0]
        assert "tokens_per_second" in results[0]

    def test_evaluate_benchmarks_graceful_without_lm_eval(self) -> None:
        """Patch import, verify no crash."""
        from src.evaluation.evaluator import ModelEvaluator

        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = MagicMock()
        evaluator.tokenizer = MagicMock()

        with patch("src.evaluation.evaluator._LM_EVAL_AVAILABLE", False):
            results = evaluator.evaluate_benchmarks(["mmlu"])
            assert "error" in results

    def test_evaluate_custom_task(self, tmp_dir: Any) -> None:
        """Mock generation, verify metrics computed."""
        import json
        import torch
        from pathlib import Path
        from src.evaluation.evaluator import ModelEvaluator

        # Create temp eval dataset
        dataset_path = Path(tmp_dir) / "eval.jsonl"
        with open(dataset_path, "w") as f:
            f.write(json.dumps({"input": "What is 2+2?", "output": "4"}) + "\n")
            f.write(json.dumps({"input": "What is 3+3?", "output": "6"}) + "\n")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.return_value = {
            "input_ids": torch.ones(1, 5, dtype=torch.long),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        mock_tokenizer.decode.return_value = "4"

        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = [torch.ones(10, dtype=torch.long)]

        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer

        results = evaluator.evaluate_custom_task(
            str(dataset_path), metrics=["exact_match", "f1_token"]
        )

        assert "exact_match" in results
        assert "f1_token" in results
        assert isinstance(results["exact_match"], float)

    def test_run_full_evaluation(self) -> None:
        """Verify all sub-evaluations called based on config."""
        from src.evaluation.evaluator import ModelEvaluator

        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model = MagicMock()
        evaluator.tokenizer = MagicMock()

        with patch.object(evaluator, "evaluate_benchmarks", return_value={"mmlu": {"acc": 0.7}}):
            with patch("src.evaluation.evaluator._LM_EVAL_AVAILABLE", True):
                results = evaluator.run_full_evaluation(
                    {"benchmarks": ["mmlu"]}
                )
                assert "benchmarks" in results
