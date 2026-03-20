"""Unit tests for evaluation.report module."""

import json
from pathlib import Path

from src.evaluation.report import ReportGenerator


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_generate_json_report(self, tmp_path: Path) -> None:
        """Test JSON report generation."""
        results = {"benchmark_a": {"accuracy": 0.95, "loss": 0.1}}
        output_path = tmp_path / "report.json"
        path = ReportGenerator.generate_json_report(results, output_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "timestamp" in data
        assert data["results"]["benchmark_a"]["accuracy"] == 0.95

    def test_generate_markdown_report(self, tmp_path: Path) -> None:
        """Test Markdown report generation."""
        results = {"benchmark_a": {"accuracy": 0.95}}
        output_path = tmp_path / "report.md"
        path = ReportGenerator.generate_markdown_report(results, output_path)
        assert path.exists()
        content = path.read_text()
        assert "# Evaluation Report" in content
        assert "benchmark_a" in content
