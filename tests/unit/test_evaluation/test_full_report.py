"""Unit tests for evaluation.report generate_full_evaluation_report."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from typing import Any

import pytest

from src.evaluation.report import ReportGenerator


@pytest.fixture()
def sample_eval_results() -> dict[str, Any]:
    """Sample evaluation results."""
    return {
        "perplexity": 12.5,
        "benchmarks": {
            "mmlu": {"accuracy": 0.75},
            "hellaswag": {"accuracy": 0.68},
        },
        "custom_domain_qa": {
            "exact_match": 0.60,
            "rouge_l": 0.45,
        },
        "generation": [
            {
                "prompt": "What is ML?",
                "generated_text": "Machine learning is...",
                "num_tokens": 50,
                "latency_ms": 200,
                "tokens_per_second": 250,
            }
        ],
        "toxicity": {
            "mean_toxicity": 0.05,
            "max_toxicity": 0.1,
            "num_flagged": 0,
        },
    }


@pytest.fixture()
def sample_config() -> dict[str, Any]:
    """Sample evaluation config."""
    return {
        "thresholds": {
            "max_perplexity": 15.0,
            "min_rouge_l": 0.3,
            "min_exact_match": 0.5,
            "max_toxicity": 0.1,
        }
    }


class TestFullEvaluationReport:
    """Tests for generate_full_evaluation_report."""

    def test_full_report_contains_all_sections(
        self,
        sample_eval_results: dict[str, Any],
        sample_config: dict[str, Any],
        tmp_dir: Path,
    ) -> None:
        """Verify section headers present."""
        output_path = str(tmp_dir / "report.md")
        report = ReportGenerator.generate_full_evaluation_report(
            sample_eval_results, sample_config, output_path=output_path
        )

        expected_sections = [
            "1. Executive Summary",
            "2. Benchmark Results",
            "3. Custom Task Results",
            "4. Generation Quality",
            "5. Perplexity Analysis",
            "6. Base Model Comparison",
            "7. Toxicity & Safety",
            "8. Recommendations",
        ]
        for section in expected_sections:
            assert f"## {section}" in report, f"Missing section: {section}"

    def test_full_report_pass_fail_thresholds(
        self,
        sample_eval_results: dict[str, Any],
        sample_config: dict[str, Any],
        tmp_dir: Path,
    ) -> None:
        """Verify pass/fail logic."""
        output_path = str(tmp_dir / "report.md")
        report = ReportGenerator.generate_full_evaluation_report(
            sample_eval_results, sample_config, output_path=output_path
        )

        # Perplexity 12.5 < 15.0 threshold => PASS
        assert "PASS" in report

    def test_full_report_without_comparison(
        self,
        sample_eval_results: dict[str, Any],
        sample_config: dict[str, Any],
        tmp_dir: Path,
    ) -> None:
        """Verify no crash when comparison=None."""
        output_path = str(tmp_dir / "report.md")
        report = ReportGenerator.generate_full_evaluation_report(
            sample_eval_results, sample_config,
            comparison=None,
            output_path=output_path,
        )

        assert "No base model comparison available." in report

    def test_full_report_saves_to_file(
        self,
        sample_eval_results: dict[str, Any],
        sample_config: dict[str, Any],
        tmp_dir: Path,
    ) -> None:
        """Verify file written."""
        output_path = str(tmp_dir / "report.md")
        report = ReportGenerator.generate_full_evaluation_report(
            sample_eval_results, sample_config, output_path=output_path
        )

        assert Path(output_path).exists()
        content = Path(output_path).read_text()
        assert content == report

    def test_full_report_with_empty_results(
        self, tmp_dir: Path
    ) -> None:
        """Verify graceful output with empty results."""
        output_path = str(tmp_dir / "report.md")
        report = ReportGenerator.generate_full_evaluation_report(
            eval_results={},
            config={},
            output_path=output_path,
        )

        assert "# Full Evaluation Report" in report
        assert "No benchmark results available." in report
