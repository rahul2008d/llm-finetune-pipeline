"""Side-by-side model comparison with statistical testing."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricComparison:
    """Comparison of a single metric between two models."""

    metric_name: str
    model_a_value: float
    model_b_value: float
    difference: float
    relative_change_pct: float
    p_value: float | None  # from bootstrap test
    winner: str  # "model_a", "model_b", or "tie"


@dataclass
class GenerationComparison:
    """Side-by-side generation example."""

    prompt: str
    model_a_output: str
    model_b_output: str


@dataclass
class ComparisonReport:
    """Complete comparison between two models."""

    model_a_name: str
    model_b_name: str
    metrics_comparison: dict[str, MetricComparison] = field(default_factory=dict)
    generation_examples: list[GenerationComparison] = field(default_factory=list)
    recommendation: str = ""


class ModelComparator:
    """Compare two models on identical evaluation data."""

    def compare(
        self,
        model_a_path: str,
        model_b_path: str,
        eval_dataset: Any,
        metrics: list[str],
        num_generation_examples: int = 20,
    ) -> ComparisonReport:
        """Full comparison of two models.

        1. Load both models via ModelEvaluator
        2. Run identical evaluation on both
        3. Compute paired bootstrap confidence intervals (1000 resamples)
        4. Generate comparison table with win/loss/tie per metric
        5. Generate side-by-side generation examples (random sample)
        6. Auto-generate recommendation text

        Args:
            model_a_path: Path to model A.
            model_b_path: Path to model B.
            eval_dataset: Evaluation dataset.
            metrics: List of metrics to compare.
            num_generation_examples: Number of side-by-side examples.

        Returns:
            ComparisonReport with detailed results.
        """
        from src.evaluation.evaluator import ModelEvaluator

        evaluator_a = ModelEvaluator(model_a_path)
        evaluator_b = ModelEvaluator(model_b_path)

        report = ComparisonReport(
            model_a_name=model_a_path,
            model_b_name=model_b_path,
        )

        # Evaluate both models
        prompts = self._extract_prompts(eval_dataset)
        references = self._extract_references(eval_dataset)

        gen_a = evaluator_a.evaluate_generation(prompts, do_sample=False)
        gen_b = evaluator_b.evaluate_generation(prompts, do_sample=False)

        preds_a = [g["generated_text"] for g in gen_a]
        preds_b = [g["generated_text"] for g in gen_b]

        # Compute metrics
        for metric_name in metrics:
            scores_a = self._compute_per_sample_metric(
                metric_name, preds_a, references
            )
            scores_b = self._compute_per_sample_metric(
                metric_name, preds_b, references
            )

            val_a = float(np.mean(scores_a)) if scores_a else 0.0
            val_b = float(np.mean(scores_b)) if scores_b else 0.0
            diff = val_b - val_a
            rel_change = (diff / abs(val_a) * 100) if val_a != 0 else 0.0

            p_value = self._bootstrap_p_value(scores_a, scores_b)

            if p_value is not None and p_value < 0.05:
                winner = "model_b" if diff > 0 else "model_a"
            else:
                winner = "tie"

            report.metrics_comparison[metric_name] = MetricComparison(
                metric_name=metric_name,
                model_a_value=val_a,
                model_b_value=val_b,
                difference=diff,
                relative_change_pct=rel_change,
                p_value=p_value,
                winner=winner,
            )

        # Generation examples
        sample_indices = list(range(len(prompts)))
        random.seed(42)
        random.shuffle(sample_indices)
        sample_indices = sample_indices[:num_generation_examples]

        for idx in sample_indices:
            report.generation_examples.append(
                GenerationComparison(
                    prompt=prompts[idx],
                    model_a_output=preds_a[idx],
                    model_b_output=preds_b[idx],
                )
            )

        # Auto-generate recommendation
        report.recommendation = self._generate_recommendation(report)

        logger.info(
            "Model comparison complete",
            model_a=model_a_path,
            model_b=model_b_path,
            num_metrics=len(metrics),
        )
        return report

    def compare_against_base(
        self,
        finetuned_path: str,
        base_model_name: str,
        eval_dataset: Any,
        metrics: list[str] | None = None,
    ) -> ComparisonReport:
        """Compare fine-tuned model against its base.

        Default metrics: perplexity, rouge_l, exact_match.
        Useful for measuring improvement from fine-tuning.

        Args:
            finetuned_path: Path to the fine-tuned model.
            base_model_name: HuggingFace model ID for the base model.
            eval_dataset: Evaluation dataset.
            metrics: Optional list of metrics (defaults to standard set).

        Returns:
            ComparisonReport comparing fine-tuned vs base.
        """
        if metrics is None:
            metrics = ["rouge_l", "exact_match"]

        return self.compare(
            model_a_path=base_model_name,
            model_b_path=finetuned_path,
            eval_dataset=eval_dataset,
            metrics=metrics,
        )

    @staticmethod
    def _bootstrap_p_value(
        scores_a: list[float],
        scores_b: list[float],
        n_resamples: int = 1000,
        seed: int = 42,
    ) -> float:
        """Compute p-value via paired bootstrap resampling.

        Args:
            scores_a: Per-sample scores for model A.
            scores_b: Per-sample scores for model B.
            n_resamples: Number of bootstrap resamples.
            seed: Random seed for reproducibility.

        Returns:
            p-value as a float between 0 and 1.
        """
        if not scores_a or not scores_b:
            return 1.0

        rng = np.random.RandomState(seed)
        n = min(len(scores_a), len(scores_b))
        arr_a = np.array(scores_a[:n])
        arr_b = np.array(scores_b[:n])

        observed_diff = float(np.mean(arr_b) - np.mean(arr_a))
        count_extreme = 0

        for _ in range(n_resamples):
            indices = rng.randint(0, n, size=n)
            boot_diff = float(np.mean(arr_b[indices]) - np.mean(arr_a[indices]))
            if observed_diff >= 0 and boot_diff <= 0:
                count_extreme += 1
            elif observed_diff < 0 and boot_diff >= 0:
                count_extreme += 1

        return count_extreme / n_resamples

    @staticmethod
    def _extract_prompts(dataset: Any) -> list[str]:
        """Extract prompts from an evaluation dataset.

        Args:
            dataset: Dataset-like object.

        Returns:
            List of prompt strings.
        """
        if hasattr(dataset, "column_names"):
            for col in ("input", "prompt", "question", "text"):
                if col in dataset.column_names:
                    return list(dataset[col])
        if isinstance(dataset, list):
            return [
                d.get("input", d.get("prompt", str(d)))
                for d in dataset
            ]
        return [str(d) for d in dataset]

    @staticmethod
    def _extract_references(dataset: Any) -> list[str]:
        """Extract references from an evaluation dataset.

        Args:
            dataset: Dataset-like object.

        Returns:
            List of reference strings.
        """
        if hasattr(dataset, "column_names"):
            for col in ("output", "reference", "answer", "target"):
                if col in dataset.column_names:
                    return list(dataset[col])
        if isinstance(dataset, list):
            return [
                d.get("output", d.get("reference", ""))
                for d in dataset
            ]
        return [""] * len(list(dataset))

    @staticmethod
    def _compute_per_sample_metric(
        metric_name: str,
        predictions: list[str],
        references: list[str],
    ) -> list[float]:
        """Compute per-sample scores for a given metric.

        Args:
            metric_name: Name of the metric.
            predictions: List of predicted texts.
            references: List of reference texts.

        Returns:
            List of per-sample float scores.
        """
        from src.evaluation.metrics import compute_f1_token_overlap

        scores: list[float] = []
        for pred, ref in zip(predictions, references):
            if metric_name == "exact_match":
                scores.append(1.0 if pred.strip().lower() == ref.strip().lower() else 0.0)
            elif metric_name == "f1_token":
                scores.append(compute_f1_token_overlap(pred, ref))
            elif metric_name == "rouge_l":
                try:
                    import evaluate

                    rouge = evaluate.load("rouge")
                    result = rouge.compute(
                        predictions=[pred], references=[ref]
                    )
                    scores.append(float(result.get("rougeL", 0.0)))
                except Exception:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    @staticmethod
    def _generate_recommendation(report: ComparisonReport) -> str:
        """Auto-generate recommendation text based on comparison results.

        Args:
            report: The comparison report with metrics.

        Returns:
            Recommendation string.
        """
        wins_a = sum(
            1 for m in report.metrics_comparison.values() if m.winner == "model_a"
        )
        wins_b = sum(
            1 for m in report.metrics_comparison.values() if m.winner == "model_b"
        )
        ties = sum(
            1 for m in report.metrics_comparison.values() if m.winner == "tie"
        )

        if wins_b > wins_a:
            return (
                f"Model B ({report.model_b_name}) is recommended. "
                f"It wins on {wins_b}/{len(report.metrics_comparison)} metrics "
                f"with {ties} ties."
            )
        elif wins_a > wins_b:
            return (
                f"Model A ({report.model_a_name}) is recommended. "
                f"It wins on {wins_a}/{len(report.metrics_comparison)} metrics "
                f"with {ties} ties."
            )
        else:
            return (
                f"No clear winner. Both models are tied on "
                f"{ties}/{len(report.metrics_comparison)} metrics. "
                "Consider additional evaluation criteria."
            )


__all__: list[str] = [
    "MetricComparison",
    "GenerationComparison",
    "ComparisonReport",
    "ModelComparator",
]
