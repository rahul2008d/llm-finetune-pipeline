"""Metrics computation for evaluating fine-tuned models."""

from typing import Any

import evaluate
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class MetricsComputer:
    """Compute evaluation metrics for model outputs."""

    @staticmethod
    def compute_perplexity(losses: list[float]) -> float:
        """Compute perplexity from a list of per-token losses.

        Args:
            losses: List of cross-entropy loss values.

        Returns:
            Perplexity score.
        """
        avg_loss = float(np.mean(losses))
        perplexity = float(np.exp(avg_loss))
        logger.info("Perplexity computed", perplexity=perplexity, avg_loss=avg_loss)
        return perplexity

    @staticmethod
    def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
        """Compute ROUGE scores between predictions and references.

        Args:
            predictions: List of predicted text strings.
            references: List of reference text strings.

        Returns:
            Dictionary of ROUGE metric scores.
        """
        rouge = evaluate.load("rouge")
        results: dict[str, float] = rouge.compute(
            predictions=predictions, references=references
        )
        logger.info("ROUGE scores computed", **results)
        return results

    @staticmethod
    def compute_bleu(predictions: list[str], references: list[list[str]]) -> dict[str, Any]:
        """Compute BLEU score between predictions and references.

        Args:
            predictions: List of predicted text strings.
            references: List of reference text string lists.

        Returns:
            Dictionary with BLEU score and related metrics.
        """
        bleu = evaluate.load("bleu")
        results: dict[str, Any] = bleu.compute(predictions=predictions, references=references)
        logger.info("BLEU score computed", bleu=results.get("bleu"))
        return results

    @staticmethod
    def compute_accuracy(predictions: list[str], references: list[str]) -> float:
        """Compute exact-match accuracy.

        Args:
            predictions: List of predicted strings.
            references: List of reference strings.

        Returns:
            Accuracy as a float between 0 and 1.
        """
        correct = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
        accuracy = correct / len(predictions) if predictions else 0.0
        logger.info("Accuracy computed", accuracy=accuracy, total=len(predictions))
        return accuracy


__all__: list[str] = ["MetricsComputer"]
