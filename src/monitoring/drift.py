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


__all__: list[str] = ["DriftDetector"]
