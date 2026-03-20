"""Unit tests for custom metrics functions in evaluation.metrics module."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    compute_batch_f1,
    compute_coherence_score,
    compute_diversity,
    compute_exact_match,
    compute_f1_token_overlap,
    compute_repetition_rate,
    compute_toxicity,
)


class TestExactMatch:
    """Tests for compute_exact_match."""

    def test_exact_match_identical_strings(self) -> None:
        """Returns 1.0 for identical strings."""
        preds = ["hello world", "foo bar"]
        refs = ["hello world", "foo bar"]
        assert compute_exact_match(preds, refs) == 1.0

    def test_exact_match_with_normalization(self) -> None:
        """'The cat' vs 'the cat' returns 1.0."""
        preds = ["The cat"]
        refs = ["the cat"]
        assert compute_exact_match(preds, refs, normalize=True) == 1.0

    def test_exact_match_empty_input(self) -> None:
        """Returns 0.0 for empty inputs."""
        assert compute_exact_match([], []) == 0.0


class TestF1TokenOverlap:
    """Tests for compute_f1_token_overlap."""

    def test_f1_token_overlap_partial(self) -> None:
        """'hello world' vs 'hello there' returns expected F1."""
        f1 = compute_f1_token_overlap("hello world", "hello there")
        # 1 common token out of 2 each: precision=0.5, recall=0.5, F1=0.5
        assert abs(f1 - 0.5) < 1e-6

    def test_f1_token_overlap_no_overlap(self) -> None:
        """Returns 0.0 when no overlap."""
        f1 = compute_f1_token_overlap("cat dog", "fish bird")
        assert f1 == 0.0

    def test_f1_token_overlap_empty(self) -> None:
        """Returns 0.0 for empty inputs."""
        assert compute_f1_token_overlap("", "") == 0.0
        assert compute_f1_token_overlap("hello", "") == 0.0
        assert compute_f1_token_overlap("", "hello") == 0.0


class TestBatchF1:
    """Tests for compute_batch_f1."""

    def test_batch_f1_averages_correctly(self) -> None:
        """Verify average of individual F1 scores."""
        preds = ["hello world", "cat dog"]
        refs = ["hello there", "cat fish"]

        batch_score = compute_batch_f1(preds, refs)
        individual_scores = [
            compute_f1_token_overlap(p, r) for p, r in zip(preds, refs)
        ]
        expected = sum(individual_scores) / len(individual_scores)
        assert abs(batch_score - expected) < 1e-6


class TestDiversity:
    """Tests for compute_diversity."""

    def test_diversity_all_same(self) -> None:
        """Repeated text returns low distinct scores."""
        texts = ["hello hello hello"] * 5
        result = compute_diversity(texts)
        assert result["distinct_1"] < 0.5

    def test_diversity_all_different(self) -> None:
        """Varied text returns high distinct scores."""
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "a completely different sentence with unique words",
            "machine learning is transforming artificial intelligence",
            "python programming enables rapid software development",
        ]
        result = compute_diversity(texts)
        assert result["distinct_1"] > 0.5

    def test_diversity_empty(self) -> None:
        """Returns all 0.0 for empty input."""
        result = compute_diversity([])
        assert result["distinct_1"] == 0.0
        assert result["distinct_2"] == 0.0
        assert result["distinct_3"] == 0.0
        assert result["self_bleu"] == 0.0


class TestToxicity:
    """Tests for compute_toxicity."""

    def test_toxicity_clean_text(self) -> None:
        """Returns low scores for clean text."""
        texts = ["Hello, how are you?", "The weather is nice today."]
        result = compute_toxicity(texts)
        assert result["mean_toxicity"] < 0.5
        assert result["num_flagged"] == 0

    def test_toxicity_handles_missing_model(self) -> None:
        """No crash without detoxify."""
        texts = ["Some normal text."]
        result = compute_toxicity(texts)
        assert "mean_toxicity" in result
        assert "max_toxicity" in result
        assert "num_flagged" in result
        assert "flagged_indices" in result


class TestRepetitionRate:
    """Tests for compute_repetition_rate."""

    def test_repetition_rate_no_repeats(self) -> None:
        """Returns low value for unique text."""
        texts = [
            "the quick brown fox",
            "a completely different sentence here",
            "machine learning algorithms work well",
        ]
        rate = compute_repetition_rate(texts)
        assert rate < 0.5

    def test_repetition_rate_high_repeats(self) -> None:
        """Returns high value for repeated text."""
        texts = ["the cat sat on the mat"] * 10
        rate = compute_repetition_rate(texts)
        assert rate > 0.5


class TestCoherenceScore:
    """Tests for compute_coherence_score."""

    def test_coherence_score_returns_float(self) -> None:
        """Verify type."""
        texts = ["This is a sentence.", "Another sentence here.", "And one more."]
        score = compute_coherence_score(texts)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestEmptyInputsGraceful:
    """Verify every function returns gracefully on empty lists."""

    def test_all_metrics_handle_empty_lists(self) -> None:
        """Every function handles empty inputs without crashing."""
        assert compute_exact_match([], []) == 0.0
        assert compute_f1_token_overlap("", "") == 0.0
        assert compute_batch_f1([], []) == 0.0
        assert compute_coherence_score([]) == 0.0

        diversity = compute_diversity([])
        assert all(v == 0.0 for v in diversity.values())

        toxicity = compute_toxicity([])
        assert toxicity["mean_toxicity"] == 0.0

        assert compute_repetition_rate([]) == 0.0
