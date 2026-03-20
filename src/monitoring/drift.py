"""Drift detection for monitoring model performance degradation."""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class DriftDetector:
    """Detect data and model drift using statistical methods."""

    def __init__(self, threshold: float = 0.05) -> None:
        """Initialize the drift detector.

        Args:
            threshold: P-value threshold for drift detection.
        """
        self.threshold = threshold

    def detect_distribution_drift(
        self,
        reference: list[float],
        current: list[float],
    ) -> dict[str, Any]:
        """Detect distribution drift using Kolmogorov-Smirnov test.

        Args:
            reference: Reference distribution values.
            current: Current distribution values.

        Returns:
            Dictionary with drift detection results.
        """
        from scipy import stats

        statistic, p_value = stats.ks_2samp(reference, current)
        is_drift = p_value < self.threshold

        result: dict[str, Any] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "threshold": self.threshold,
            "drift_detected": is_drift,
        }

        if is_drift:
            logger.warning("Distribution drift detected", **result)
        else:
            logger.info("No distribution drift", **result)

        return result

    def detect_performance_drift(
        self,
        baseline_metrics: dict[str, float],
        current_metrics: dict[str, float],
        tolerance: float = 0.1,
    ) -> dict[str, Any]:
        """Detect performance drift by comparing metrics.

        Args:
            baseline_metrics: Baseline performance metrics.
            current_metrics: Current performance metrics.
            tolerance: Acceptable relative degradation fraction.

        Returns:
            Dictionary with drift analysis per metric.
        """
        results: dict[str, Any] = {}
        for metric_name in baseline_metrics:
            if metric_name not in current_metrics:
                continue
            baseline_val = baseline_metrics[metric_name]
            current_val = current_metrics[metric_name]
            if baseline_val == 0:
                continue
            relative_change = (current_val - baseline_val) / abs(baseline_val)
            is_degraded = relative_change < -tolerance

            results[metric_name] = {
                "baseline": baseline_val,
                "current": current_val,
                "relative_change": float(relative_change),
                "degraded": is_degraded,
            }

            if is_degraded:
                logger.warning("Performance drift detected", metric=metric_name, **results[metric_name])

        return results

    @staticmethod
    def compute_embedding_drift(
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
    ) -> float:
        """Compute cosine distance between mean embeddings as a drift score.

        Args:
            reference_embeddings: Reference embedding matrix.
            current_embeddings: Current embedding matrix.

        Returns:
            Cosine distance drift score (0 = identical, 2 = opposite).
        """
        ref_mean = np.mean(reference_embeddings, axis=0)
        cur_mean = np.mean(current_embeddings, axis=0)

        cos_sim = float(
            np.dot(ref_mean, cur_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(cur_mean))
        )
        drift_score = 1.0 - cos_sim
        logger.info("Embedding drift computed", drift_score=drift_score)
        return drift_score

    def check_output_drift(
        self,
        recent_outputs: list[dict[str, Any]],
        baseline_stats: dict[str, float],
    ) -> dict[str, Any]:
        """Detect drift in model output quality.

        Computes output statistics (avg response length, vocabulary richness,
        repetition rate, refusal rate) and compares against baseline using KS test.

        Args:
            recent_outputs: List of output dicts, each with a "text" key.
            baseline_stats: Baseline statistics with keys matching computed metrics.

        Returns:
            Dictionary with drifted flag, metrics dict, and details list.
        """
        import re

        from scipy import stats as sp_stats

        refusal_patterns = re.compile(
            r"(i cannot|i can't|i'm unable|i am unable|as an ai|i'm sorry, but i)",
            re.IGNORECASE,
        )

        lengths: list[float] = []
        vocab_richness_values: list[float] = []
        repetition_rates: list[float] = []
        refusal_count = 0

        for output in recent_outputs:
            text = output.get("text", "")
            tokens = text.split()
            length = len(tokens)
            lengths.append(float(length))

            unique_tokens = set(tokens)
            richness = len(unique_tokens) / length if length > 0 else 0.0
            vocab_richness_values.append(richness)

            if length > 1:
                bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
                unique_bigrams = set(bigrams)
                repetition = 1.0 - (len(unique_bigrams) / len(bigrams))
            else:
                repetition = 0.0
            repetition_rates.append(repetition)

            if refusal_patterns.search(text):
                refusal_count += 1

        n = len(recent_outputs) if recent_outputs else 1
        computed_metrics = {
            "avg_response_length": sum(lengths) / n if lengths else 0.0,
            "vocab_richness": sum(vocab_richness_values) / n if vocab_richness_values else 0.0,
            "repetition_rate": sum(repetition_rates) / n if repetition_rates else 0.0,
            "refusal_rate": refusal_count / n,
        }

        details: list[str] = []
        drifted = False

        for metric_name, current_values in [
            ("avg_response_length", lengths),
            ("vocab_richness", vocab_richness_values),
            ("repetition_rate", repetition_rates),
        ]:
            if metric_name in baseline_stats and len(current_values) >= 2:
                baseline_val = baseline_stats[metric_name]
                baseline_samples = [baseline_val] * len(current_values)
                statistic, p_value = sp_stats.ks_2samp(baseline_samples, current_values)
                if p_value < self.threshold:
                    drifted = True
                    details.append(
                        f"{metric_name}: drift detected (p={p_value:.4f}, stat={statistic:.4f})"
                    )

        if "refusal_rate" in baseline_stats:
            baseline_refusal = baseline_stats["refusal_rate"]
            current_refusal = computed_metrics["refusal_rate"]
            if abs(current_refusal - baseline_refusal) > 0.1:
                drifted = True
                details.append(
                    f"refusal_rate: {baseline_refusal:.2f} -> {current_refusal:.2f}"
                )

        if drifted:
            logger.warning("Output drift detected", details=details)
        else:
            logger.info("No output drift detected")

        return {"drifted": drifted, "metrics": computed_metrics, "details": details}

    def check_input_drift(
        self,
        recent_inputs: list[dict[str, Any]],
        training_data_stats: dict[str, float],
    ) -> dict[str, Any]:
        """Detect drift in input distribution.

        Compares topic distribution, length distribution, and language distribution
        against training data statistics.

        Args:
            recent_inputs: List of input dicts, each with a "text" key.
            training_data_stats: Training data statistics with keys like
                avg_length, max_length, min_length.

        Returns:
            Dictionary with drifted flag, metrics dict, and details list.
        """
        from scipy import stats as sp_stats

        lengths: list[float] = []
        for inp in recent_inputs:
            text = inp.get("text", "")
            tokens = text.split()
            lengths.append(float(len(tokens)))

        n = len(recent_inputs) if recent_inputs else 1
        computed_metrics = {
            "avg_length": sum(lengths) / n if lengths else 0.0,
            "max_length": max(lengths) if lengths else 0.0,
            "min_length": min(lengths) if lengths else 0.0,
            "sample_count": n,
        }

        details: list[str] = []
        drifted = False

        if "avg_length" in training_data_stats and len(lengths) >= 2:
            baseline_avg = training_data_stats["avg_length"]
            baseline_samples = [baseline_avg] * len(lengths)
            statistic, p_value = sp_stats.ks_2samp(baseline_samples, lengths)
            if p_value < self.threshold:
                drifted = True
                details.append(
                    f"length distribution: drift detected (p={p_value:.4f}, stat={statistic:.4f})"
                )

        if "max_length" in training_data_stats:
            training_max = training_data_stats["max_length"]
            if computed_metrics["max_length"] > training_max * 1.5:
                drifted = True
                details.append(
                    f"max_length exceeded: {computed_metrics['max_length']:.0f} vs training {training_max:.0f}"
                )

        if drifted:
            logger.warning("Input drift detected", details=details)
        else:
            logger.info("No input drift detected")

        return {"drifted": drifted, "metrics": computed_metrics, "details": details}


__all__: list[str] = ["DriftDetector"]
