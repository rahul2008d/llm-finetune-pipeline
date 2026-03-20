"""Evaluation pipeline: metrics computation, benchmarks, and report generation."""

from src.evaluation.benchmarks import BenchmarkRunner
from src.evaluation.comparator import ComparisonReport, ModelComparator
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import MetricsComputer
from src.evaluation.report import ReportGenerator

__all__: list[str] = [
    "MetricsComputer",
    "BenchmarkRunner",
    "ReportGenerator",
    "ModelEvaluator",
    "ModelComparator",
    "ComparisonReport",
]
