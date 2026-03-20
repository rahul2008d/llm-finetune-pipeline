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

    def test_check_output_drift_no_drift(self) -> None:
        """Test output drift detection with similar distributions (no drift)."""
        detector = DriftDetector(threshold=0.05)
        recent_outputs = [
            {"text": "This is a normal response with good content."},
            {"text": "Another reasonable output with varied vocabulary."},
            {"text": "The model generates helpful and diverse answers."},
            {"text": "Each response contains unique and relevant information."},
        ]
        baseline_stats = {
            "avg_response_length": 7.5,
            "vocab_richness": 0.9,
            "repetition_rate": 0.1,
            "refusal_rate": 0.0,
        }
        result = detector.check_output_drift(recent_outputs, baseline_stats)
        assert "drifted" in result
        assert "metrics" in result
        assert "details" in result
        assert isinstance(result["metrics"]["avg_response_length"], float)
        assert isinstance(result["metrics"]["refusal_rate"], float)

    def test_check_output_drift_detected(self) -> None:
        """Test output drift detection with very different distributions."""
        detector = DriftDetector(threshold=0.05)
        recent_outputs = [
            {"text": "I cannot help with that request."},
            {"text": "I'm sorry, but I am unable to assist."},
            {"text": "As an AI, I cannot provide that information."},
            {"text": "I can't do that."},
            {"text": "I'm unable to help."},
        ]
        baseline_stats = {
            "avg_response_length": 50.0,
            "vocab_richness": 0.85,
            "repetition_rate": 0.05,
            "refusal_rate": 0.01,
        }
        result = detector.check_output_drift(recent_outputs, baseline_stats)
        assert result["drifted"] is True
        assert len(result["details"]) > 0
        assert result["metrics"]["refusal_rate"] == 1.0

    def test_check_input_drift(self) -> None:
        """Test input drift comparison works correctly."""
        detector = DriftDetector(threshold=0.05)
        recent_inputs = [
            {"text": "Short query"},
            {"text": "Another brief question here"},
            {"text": "Simple input for testing"},
            {"text": "Basic request"},
        ]
        training_data_stats = {
            "avg_length": 3.0,
            "max_length": 10.0,
            "min_length": 1.0,
        }
        result = detector.check_input_drift(recent_inputs, training_data_stats)
        assert "drifted" in result
        assert "metrics" in result
        assert "details" in result
        assert result["metrics"]["sample_count"] == 4
        assert result["metrics"]["avg_length"] > 0
