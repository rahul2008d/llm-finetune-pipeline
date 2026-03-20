"""Benchmark runners for standard LLM evaluation tasks."""

from typing import Any

import structlog
from datasets import Dataset

from src.evaluation.metrics import MetricsComputer

logger = structlog.get_logger(__name__)


class BenchmarkRunner:
    """Run standardized benchmarks against a fine-tuned model."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        """Initialize the benchmark runner.

        Args:
            model: The fine-tuned model to evaluate.
            tokenizer: Associated tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.metrics_computer = MetricsComputer()

    def run_benchmark(
        self,
        benchmark_name: str,
        dataset: Dataset,
        input_column: str = "input",
        reference_column: str = "output",
        max_new_tokens: int = 256,
    ) -> dict[str, Any]:
        """Run a benchmark evaluation on the model.

        Args:
            benchmark_name: Name of the benchmark for logging.
            dataset: Evaluation dataset.
            input_column: Column with input prompts.
            reference_column: Column with reference outputs.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Dictionary of benchmark results.
        """
        logger.info("Running benchmark", benchmark=benchmark_name, num_samples=len(dataset))

        predictions: list[str] = []
        references: list[str] = list(dataset[reference_column])

        for row in dataset:
            inputs = self.tokenizer(
                row[input_column], return_tensors="pt", truncation=True, max_length=2048
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(decoded)

        results: dict[str, Any] = {
            "benchmark": benchmark_name,
            "num_samples": len(dataset),
            "accuracy": MetricsComputer.compute_accuracy(predictions, references),
            "rouge": MetricsComputer.compute_rouge(predictions, references),
        }

        logger.info("Benchmark complete", benchmark=benchmark_name, results=results)
        return results

    def run_all(
        self,
        benchmarks: dict[str, Dataset],
        input_column: str = "input",
        reference_column: str = "output",
    ) -> dict[str, dict[str, Any]]:
        """Run all specified benchmarks.

        Args:
            benchmarks: Mapping of benchmark names to datasets.
            input_column: Column with input prompts.
            reference_column: Column with reference outputs.

        Returns:
            Dictionary mapping benchmark names to their results.
        """
        all_results: dict[str, dict[str, Any]] = {}
        for name, dataset in benchmarks.items():
            all_results[name] = self.run_benchmark(
                name, dataset, input_column, reference_column
            )
        return all_results


__all__: list[str] = ["BenchmarkRunner"]
