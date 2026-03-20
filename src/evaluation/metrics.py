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


# ═══════════════════════════════════════════════════════════════
# Standalone metric functions
# ═══════════════════════════════════════════════════════════════


def compute_exact_match(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
) -> float:
    """Compute exact match accuracy.

    If normalize=True: lowercase, strip whitespace, remove articles (a, an, the).

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.
        normalize: Whether to normalize text before comparison.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not predictions or not references:
        logger.warning("Empty inputs for exact_match — returning 0.0")
        return 0.0

    import re

    def _normalize(text: str) -> str:
        text = text.lower().strip()
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Collapse whitespace
        text = " ".join(text.split())
        return text

    matches = 0
    for pred, ref in zip(predictions, references):
        if normalize:
            pred = _normalize(pred)
            ref = _normalize(ref)
        if pred == ref:
            matches += 1

    score = matches / len(predictions)
    logger.info("Exact match computed", score=score, total=len(predictions))
    return score


def compute_f1_token_overlap(prediction: str, reference: str) -> float:
    """Compute token-level F1 score (SQuAD-style).

    1. Tokenize both strings by whitespace
    2. Compute precision = matching_tokens / prediction_tokens
    3. Compute recall = matching_tokens / reference_tokens
    4. F1 = 2 * precision * recall / (precision + recall)

    Args:
        prediction: Predicted text.
        reference: Reference text.

    Returns:
        F1 score as float. Returns 0.0 for empty inputs.
    """
    pred_tokens = prediction.strip().split()
    ref_tokens = reference.strip().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_batch_f1(predictions: list[str], references: list[str]) -> float:
    """Average F1 across a batch of prediction/reference pairs.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.

    Returns:
        Mean F1 score.
    """
    if not predictions or not references:
        logger.warning("Empty inputs for batch_f1 — returning 0.0")
        return 0.0

    scores = [
        compute_f1_token_overlap(p, r)
        for p, r in zip(predictions, references)
    ]
    return float(np.mean(scores))


def compute_coherence_score(
    texts: list[str],
    tokenizer: Any = None,
) -> float:
    """Measure internal consistency of generated texts.

    Based on sentence-level perplexity variance.
    Lower variance = more coherent.

    Args:
        texts: List of generated text strings.
        tokenizer: Optional tokenizer for computing perplexity-based coherence.

    Returns:
        Normalized score between 0.0 and 1.0.
    """
    if not texts:
        logger.warning("Empty texts for coherence_score — returning 0.0")
        return 0.0

    if tokenizer is None:
        # Fallback: use sentence length variance as proxy
        sentence_lengths = [len(t.split()) for t in texts]
        if len(sentence_lengths) < 2:
            return 1.0
        variance = float(np.var(sentence_lengths))
        mean_len = float(np.mean(sentence_lengths))
        if mean_len == 0:
            return 0.0
        # Coefficient of variation, inverted and clamped
        cv = (variance ** 0.5) / mean_len
        score = max(0.0, min(1.0, 1.0 - cv))
        return score

    # With tokenizer: compute token-level entropy variance
    entropies: list[float] = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) > 1:
            unique_ratio = len(set(tokens)) / len(tokens)
            entropies.append(unique_ratio)
    if not entropies:
        return 0.0
    variance = float(np.var(entropies))
    score = max(0.0, min(1.0, 1.0 - variance))
    return score


def compute_diversity(texts: list[str]) -> dict[str, float]:
    """Compute lexical diversity metrics.

    Args:
        texts: List of generated text strings.

    Returns:
        Dictionary with distinct_1, distinct_2, distinct_3, self_bleu.
    """
    if not texts:
        logger.warning("Empty texts for diversity — returning all 0.0")
        return {
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "distinct_3": 0.0,
            "self_bleu": 0.0,
        }

    all_unigrams: list[str] = []
    all_bigrams: list[tuple[str, str]] = []
    all_trigrams: list[tuple[str, str, str]] = []

    for text in texts:
        tokens = text.lower().split()
        all_unigrams.extend(tokens)
        all_bigrams.extend(zip(tokens, tokens[1:]))
        all_trigrams.extend(zip(tokens, tokens[1:], tokens[2:]))

    distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0.0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    distinct_3 = len(set(all_trigrams)) / len(all_trigrams) if all_trigrams else 0.0

    # Self-BLEU: average BLEU of each text against all others
    self_bleu = 0.0
    if len(texts) > 1:
        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

            smoother = SmoothingFunction().method1
            bleu_scores: list[float] = []
            for i, text in enumerate(texts):
                hypothesis = text.lower().split()
                references_list = [
                    t.lower().split() for j, t in enumerate(texts) if j != i
                ]
                score = sentence_bleu(
                    references_list, hypothesis, smoothing_function=smoother
                )
                bleu_scores.append(score)
            self_bleu = float(np.mean(bleu_scores))
        except ImportError:
            logger.info("nltk not available — skipping self-BLEU computation")
            self_bleu = 0.0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "distinct_3": distinct_3,
        "self_bleu": self_bleu,
    }


def compute_toxicity(texts: list[str]) -> dict[str, Any]:
    """Score texts for toxicity.

    Uses a lightweight keyword-based approach as default.
    If detoxify is available, uses it instead.

    Args:
        texts: List of text strings to score.

    Returns:
        Dictionary with mean_toxicity, max_toxicity, num_flagged, flagged_indices.
    """
    if not texts:
        logger.warning("Empty texts for toxicity — returning zeros")
        return {
            "mean_toxicity": 0.0,
            "max_toxicity": 0.0,
            "num_flagged": 0,
            "flagged_indices": [],
        }

    try:
        from detoxify import Detoxify

        model = Detoxify("original")
        results = model.predict(texts)
        scores = results.get("toxicity", [0.0] * len(texts))
        flagged_threshold = 0.5
        flagged_indices = [i for i, s in enumerate(scores) if s >= flagged_threshold]
        return {
            "mean_toxicity": float(np.mean(scores)),
            "max_toxicity": float(np.max(scores)),
            "num_flagged": len(flagged_indices),
            "flagged_indices": flagged_indices,
        }
    except ImportError:
        logger.info("detoxify not available — using keyword-based toxicity scoring")

    # Keyword-based fallback
    _toxic_keywords = {
        "hate", "kill", "die", "stupid", "idiot", "damn", "hell",
        "violent", "attack", "abuse", "threat", "racist", "sexist",
    }
    scores: list[float] = []
    flagged_indices: list[int] = []

    for i, text in enumerate(texts):
        words = set(text.lower().split())
        matches = words & _toxic_keywords
        score = min(len(matches) / 5.0, 1.0)  # Scale: 5+ keywords = max
        scores.append(score)
        if score >= 0.5:
            flagged_indices.append(i)

    return {
        "mean_toxicity": float(np.mean(scores)),
        "max_toxicity": float(np.max(scores)) if scores else 0.0,
        "num_flagged": len(flagged_indices),
        "flagged_indices": flagged_indices,
    }


def compute_repetition_rate(texts: list[str], n: int = 3) -> float:
    """Compute the rate of repeated n-grams across generated texts.

    Higher = more repetitive.

    Args:
        texts: List of generated text strings.
        n: N-gram size.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not texts:
        logger.warning("Empty texts for repetition_rate — returning 0.0")
        return 0.0

    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    repetition = 1.0 - (unique_ngrams / total_ngrams)
    return repetition


__all__: list[str] = [
    "MetricsComputer",
    "compute_exact_match",
    "compute_f1_token_overlap",
    "compute_batch_f1",
    "compute_coherence_score",
    "compute_diversity",
    "compute_toxicity",
    "compute_repetition_rate",
]
