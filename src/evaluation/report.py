"""Report generation for evaluation results."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ReportGenerator:
    """Generate evaluation reports in JSON and Markdown formats."""

    @staticmethod
    def generate_json_report(
        results: dict[str, Any],
        output_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Generate a JSON evaluation report.

        Args:
            results: Evaluation results dictionary.
            output_path: Path to write the JSON report.
            metadata: Optional metadata to include in the report.

        Returns:
            Path to the generated report.
        """
        report = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": metadata or {},
            "results": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("JSON report generated", path=str(output_path))
        return output_path

    @staticmethod
    def generate_markdown_report(
        results: dict[str, Any],
        output_path: Path,
        title: str = "Evaluation Report",
    ) -> Path:
        """Generate a Markdown evaluation report.

        Args:
            results: Evaluation results dictionary.
            output_path: Path to write the Markdown report.
            title: Report title.

        Returns:
            Path to the generated report.
        """
        lines = [
            f"# {title}",
            "",
            f"**Generated**: {datetime.now(tz=timezone.utc).isoformat()}",
            "",
            "## Results",
            "",
        ]

        for benchmark, metrics in results.items():
            lines.append(f"### {benchmark}")
            lines.append("")
            if isinstance(metrics, dict):
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                for metric_name, value in metrics.items():
                    lines.append(f"| {metric_name} | {value} |")
            else:
                lines.append(f"Result: {metrics}")
            lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        logger.info("Markdown report generated", path=str(output_path))
        return output_path


__all__: list[str] = ["ReportGenerator"]
