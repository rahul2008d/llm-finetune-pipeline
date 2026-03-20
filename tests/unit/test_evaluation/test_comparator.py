"""Unit tests for evaluation.comparator module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import numpy as np
import pytest

from src.evaluation.comparator import (
    ComparisonReport,
    GenerationComparison,
    MetricComparison,
    ModelComparator,
)


class TestModelComparator:
    """Tests for ModelComparator."""

    @patch("src.evaluation.evaluator.ModelEvaluator")
    def test_compare_returns_report(self, mock_evaluator_cls: MagicMock) -> None:
        """Mock both models, verify ComparisonReport structure."""
        # Setup mock evaluators
        mock_eval_a = MagicMock()
        mock_eval_b = MagicMock()
        mock_evaluator_cls.side_effect = [mock_eval_a, mock_eval_b]

        mock_eval_a.evaluate_generation.return_value = [
            {"generated_text": "answer a", "num_tokens": 5, "latency_ms": 100, "tokens_per_second": 50}
        ]
        mock_eval_b.evaluate_generation.return_value = [
            {"generated_text": "answer b", "num_tokens": 5, "latency_ms": 100, "tokens_per_second": 50}
        ]

        dataset = [{"input": "test prompt", "output": "expected"}]

        comparator = ModelComparator()
        report = comparator.compare(
            model_a_path="/path/a",
            model_b_path="/path/b",
            eval_dataset=dataset,
            metrics=["exact_match"],
            num_generation_examples=1,
        )

        assert isinstance(report, ComparisonReport)
        assert report.model_a_name == "/path/a"
        assert report.model_b_name == "/path/b"
        assert "exact_match" in report.metrics_comparison

    def test_bootstrap_p_value(self) -> None:
        """Verify returns float between 0 and 1."""
        scores_a = [0.5, 0.6, 0.7, 0.8, 0.9]
        scores_b = [0.6, 0.7, 0.8, 0.9, 1.0]

        p_value = ModelComparator._bootstrap_p_value(scores_a, scores_b)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0

    @patch("src.evaluation.evaluator.ModelEvaluator")
    def test_compare_against_base(self, mock_evaluator_cls: MagicMock) -> None:
        """Verify it calls compare with correct args."""
        mock_eval_a = MagicMock()
        mock_eval_b = MagicMock()
        mock_evaluator_cls.side_effect = [mock_eval_a, mock_eval_b]

        mock_eval_a.evaluate_generation.return_value = [
            {"generated_text": "base output", "num_tokens": 5, "latency_ms": 100, "tokens_per_second": 50}
        ]
        mock_eval_b.evaluate_generation.return_value = [
            {"generated_text": "finetuned output", "num_tokens": 5, "latency_ms": 100, "tokens_per_second": 50}
        ]

        dataset = [{"input": "test", "output": "ref"}]
        comparator = ModelComparator()

        report = comparator.compare_against_base(
            finetuned_path="/path/finetuned",
            base_model_name="base-model",
            eval_dataset=dataset,
        )
        assert isinstance(report, ComparisonReport)

    @patch("src.evaluation.evaluator.ModelEvaluator")
    def test_generation_examples_count(self, mock_evaluator_cls: MagicMock) -> None:
        """Verify correct number of examples."""
        mock_eval_a = MagicMock()
        mock_eval_b = MagicMock()
        mock_evaluator_cls.side_effect = [mock_eval_a, mock_eval_b]

        prompts_data = [
            {"input": f"prompt {i}", "output": f"ref {i}"} for i in range(10)
        ]
        gen_results = [
            {"generated_text": f"gen {i}", "num_tokens": 5, "latency_ms": 100, "tokens_per_second": 50}
            for i in range(10)
        ]
        mock_eval_a.evaluate_generation.return_value = gen_results
        mock_eval_b.evaluate_generation.return_value = gen_results

        comparator = ModelComparator()
        report = comparator.compare(
            model_a_path="/a",
            model_b_path="/b",
            eval_dataset=prompts_data,
            metrics=["exact_match"],
            num_generation_examples=5,
        )

        assert len(report.generation_examples) == 5

    @patch("src.evaluation.evaluator.ModelEvaluator")
    def test_recommendation_generated(self, mock_evaluator_cls: MagicMock) -> None:
        """Verify recommendation string is non-empty."""
        mock_eval_a = MagicMock()
        mock_eval_b = MagicMock()
        mock_evaluator_cls.side_effect = [mock_eval_a, mock_eval_b]

        mock_eval_a.evaluate_generation.return_value = [
            {"generated_text": "a", "num_tokens": 1, "latency_ms": 10, "tokens_per_second": 100}
        ]
        mock_eval_b.evaluate_generation.return_value = [
            {"generated_text": "b", "num_tokens": 1, "latency_ms": 10, "tokens_per_second": 100}
        ]

        dataset = [{"input": "test", "output": "ref"}]
        comparator = ModelComparator()
        report = comparator.compare(
            model_a_path="/a",
            model_b_path="/b",
            eval_dataset=dataset,
            metrics=["exact_match"],
        )

        assert isinstance(report.recommendation, str)
        assert len(report.recommendation) > 0
