"""Unit tests for monitoring.drift module."""

from src.monitoring.drift import DriftDetector


class TestDriftDetector:
    """Tests for DriftDetector."""

    def test_no_performance_drift(self) -> None:
        """Test when metrics are within tolerance."""
        detector = DriftDetector()
        baseline = {"accuracy": 0.95, "loss": 0.1}
        current = {"accuracy": 0.94, "loss": 0.11}
        results = detector.detect_performance_drift(baseline, current, tolerance=0.1)
        assert not results["accuracy"]["degraded"]

    def test_performance_drift_detected(self) -> None:
        """Test when significant degradation is detected."""
        detector = DriftDetector()
        baseline = {"accuracy": 0.95}
        current = {"accuracy": 0.50}
        results = detector.detect_performance_drift(baseline, current, tolerance=0.1)
        assert results["accuracy"]["degraded"]
