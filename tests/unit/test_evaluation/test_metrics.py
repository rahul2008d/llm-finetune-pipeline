"""Unit tests for evaluation.metrics module."""

from src.evaluation.metrics import MetricsComputer


class TestMetricsComputer:
    """Tests for MetricsComputer."""

    def test_compute_perplexity(self) -> None:
        """Test perplexity computation."""
        losses = [1.0, 2.0, 3.0]
        perplexity = MetricsComputer.compute_perplexity(losses)
        assert perplexity > 0

    def test_compute_accuracy_perfect(self) -> None:
        """Test accuracy with perfect predictions."""
        preds = ["a", "b", "c"]
        refs = ["a", "b", "c"]
        accuracy = MetricsComputer.compute_accuracy(preds, refs)
        assert accuracy == 1.0

    def test_compute_accuracy_zero(self) -> None:
        """Test accuracy with no correct predictions."""
        preds = ["x", "y", "z"]
        refs = ["a", "b", "c"]
        accuracy = MetricsComputer.compute_accuracy(preds, refs)
        assert accuracy == 0.0

    def test_compute_accuracy_empty(self) -> None:
        """Test accuracy with empty inputs."""
        accuracy = MetricsComputer.compute_accuracy([], [])
        assert accuracy == 0.0
