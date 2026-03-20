"""Report generation for evaluation results."""

from __future__ import annotations

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

    @staticmethod
    def generate_full_evaluation_report(
        eval_results: dict[str, Any],
        config: dict[str, Any],
        comparison: Any | None = None,
        output_path: str = "results/eval_report.md",
    ) -> str:
        """Generate comprehensive Markdown evaluation report.

        Sections:
        1. Executive Summary — pass/fail against thresholds from config
        2. Benchmark Results — table with scores per benchmark
        3. Custom Task Results — table per custom task (if any)
        4. Generation Quality — sample outputs with quality scores
        5. Perplexity Analysis — perplexity score with context
        6. Base Model Comparison — ComparisonReport table (if provided)
        7. Toxicity & Safety — toxicity scores summary
        8. Recommendations — auto-generated based on results vs thresholds

        Save to output_path (local or S3).
        Log to MLflow as artifact if available.

        Args:
            eval_results: Dictionary of all evaluation results.
            config: Evaluation configuration dictionary.
            comparison: Optional ComparisonReport object.
            output_path: Path to save the report.

        Returns:
            The Markdown string.
        """
        thresholds = config.get("thresholds", {})
        lines: list[str] = []

        # Title
        lines.append("# Full Evaluation Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now(tz=timezone.utc).isoformat()}")
        lines.append("")

        # 1. Executive Summary
        lines.append("## 1. Executive Summary")
        lines.append("")
        pass_fail: list[str] = []
        perplexity = eval_results.get("perplexity")
        max_ppl = thresholds.get("max_perplexity")
        if perplexity is not None and max_ppl is not None:
            status = "PASS" if perplexity <= max_ppl else "FAIL"
            pass_fail.append(f"- Perplexity: {perplexity:.2f} (threshold: {max_ppl}) — **{status}**")

        # Check custom task thresholds
        for key, value in eval_results.items():
            if isinstance(value, dict):
                for metric_name, metric_val in value.items():
                    if not isinstance(metric_val, (int, float)):
                        continue
                    threshold_key = f"min_{metric_name}"
                    max_key = f"max_{metric_name}"
                    if threshold_key in thresholds:
                        status = "PASS" if metric_val >= thresholds[threshold_key] else "FAIL"
                        pass_fail.append(
                            f"- {key}/{metric_name}: {metric_val:.4f} "
                            f"(threshold: {thresholds[threshold_key]}) — **{status}**"
                        )
                    elif max_key in thresholds:
                        status = "PASS" if metric_val <= thresholds[max_key] else "FAIL"
                        pass_fail.append(
                            f"- {key}/{metric_name}: {metric_val:.4f} "
                            f"(threshold: {thresholds[max_key]}) — **{status}**"
                        )

        if pass_fail:
            lines.extend(pass_fail)
        else:
            lines.append("No threshold checks configured or no matching results.")
        lines.append("")

        # 2. Benchmark Results
        lines.append("## 2. Benchmark Results")
        lines.append("")
        benchmarks = eval_results.get("benchmarks", {})
        if benchmarks and not isinstance(benchmarks, str):
            lines.append("| Benchmark | Metric | Score |")
            lines.append("|-----------|--------|-------|")
            for bm_name, bm_scores in benchmarks.items():
                if isinstance(bm_scores, dict):
                    for metric_name, score in bm_scores.items():
                        if isinstance(score, (int, float)):
                            lines.append(f"| {bm_name} | {metric_name} | {score:.4f} |")
                else:
                    lines.append(f"| {bm_name} | score | {bm_scores} |")
        else:
            lines.append("No benchmark results available.")
        lines.append("")

        # 3. Custom Task Results
        lines.append("## 3. Custom Task Results")
        lines.append("")
        custom_tasks_found = False
        for key, value in eval_results.items():
            if key.startswith("custom_") and isinstance(value, dict):
                custom_tasks_found = True
                lines.append(f"### {key}")
                lines.append("")
                lines.append("| Metric | Score |")
                lines.append("|--------|-------|")
                for metric_name, score in value.items():
                    lines.append(f"| {metric_name} | {score} |")
                lines.append("")
        if not custom_tasks_found:
            lines.append("No custom task results available.")
            lines.append("")

        # 4. Generation Quality
        lines.append("## 4. Generation Quality")
        lines.append("")
        generation = eval_results.get("generation", [])
        if generation:
            for i, gen in enumerate(generation[:5]):  # Show first 5
                lines.append(f"### Example {i + 1}")
                lines.append("")
                lines.append(f"**Prompt**: {gen.get('prompt', 'N/A')}")
                lines.append("")
                lines.append(f"**Output**: {gen.get('generated_text', 'N/A')}")
                lines.append("")
                lines.append(
                    f"Tokens: {gen.get('num_tokens', 'N/A')} | "
                    f"Latency: {gen.get('latency_ms', 'N/A')}ms | "
                    f"Speed: {gen.get('tokens_per_second', 'N/A')} tok/s"
                )
                lines.append("")
        else:
            lines.append("No generation results available.")
            lines.append("")

        # 5. Perplexity Analysis
        lines.append("## 5. Perplexity Analysis")
        lines.append("")
        if perplexity is not None:
            lines.append(f"**Perplexity**: {perplexity:.4f}")
            lines.append("")
            if max_ppl:
                if perplexity <= max_ppl:
                    lines.append(f"Within acceptable range (threshold: {max_ppl}).")
                else:
                    lines.append(
                        f"**Warning**: Exceeds threshold of {max_ppl}. "
                        "Model may require additional training."
                    )
        else:
            lines.append("Perplexity evaluation was not performed.")
        lines.append("")

        # 6. Base Model Comparison
        lines.append("## 6. Base Model Comparison")
        lines.append("")
        if comparison is not None:
            lines.append(
                f"Comparing **{comparison.model_a_name}** (A) vs "
                f"**{comparison.model_b_name}** (B)"
            )
            lines.append("")
            lines.append("| Metric | Model A | Model B | Diff | Change % | Winner |")
            lines.append("|--------|---------|---------|------|----------|--------|")
            for mc in comparison.metrics_comparison.values():
                lines.append(
                    f"| {mc.metric_name} | {mc.model_a_value:.4f} | "
                    f"{mc.model_b_value:.4f} | {mc.difference:+.4f} | "
                    f"{mc.relative_change_pct:+.1f}% | {mc.winner} |"
                )
            lines.append("")
            lines.append(f"**Recommendation**: {comparison.recommendation}")
        else:
            lines.append("No base model comparison available.")
        lines.append("")

        # 7. Toxicity & Safety
        lines.append("## 7. Toxicity & Safety")
        lines.append("")
        toxicity = eval_results.get("toxicity", {})
        if toxicity:
            lines.append(f"- **Mean Toxicity**: {toxicity.get('mean_toxicity', 'N/A')}")
            lines.append(f"- **Max Toxicity**: {toxicity.get('max_toxicity', 'N/A')}")
            lines.append(f"- **Flagged Samples**: {toxicity.get('num_flagged', 0)}")
        else:
            lines.append("No toxicity analysis performed.")
        lines.append("")

        # 8. Recommendations
        lines.append("## 8. Recommendations")
        lines.append("")
        recommendations: list[str] = []
        if perplexity is not None and max_ppl is not None and perplexity > max_ppl:
            recommendations.append(
                "- Perplexity exceeds threshold. Consider additional training epochs "
                "or hyperparameter tuning."
            )
        if toxicity and toxicity.get("mean_toxicity", 0) > thresholds.get("max_toxicity", 0.1):
            recommendations.append(
                "- Toxicity levels are elevated. Review training data for harmful content."
            )
        if not recommendations:
            recommendations.append("- All metrics within acceptable thresholds. Model is ready for deployment.")
        lines.extend(recommendations)
        lines.append("")

        report_text = "\n".join(lines)

        # Save to file
        if output_path.startswith("s3://"):
            import boto3

            parts = output_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else "eval_report.md"
            s3 = boto3.client("s3")
            s3.put_object(Bucket=bucket, Key=key, Body=report_text.encode("utf-8"))
        else:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(report_text, encoding="utf-8")

        # Log to MLflow
        try:
            import mlflow

            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, prefix="eval_report_"
            ) as f:
                f.write(report_text)
                mlflow.log_artifact(f.name, "evaluation_reports")
        except ImportError:
            pass
        except Exception:
            logger.warning("Failed to log report to MLflow", exc_info=True)

        logger.info("Full evaluation report generated", path=output_path)
        return report_text


__all__: list[str] = ["ReportGenerator"]
