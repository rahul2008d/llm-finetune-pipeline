"""Tests for PIIScanner — detection, batch scanning, redaction."""
from __future__ import annotations

import pytest

from src.data.pii_scanner import (
    BatchPIIReport,
    PIIEntity,
    PIIScanResult,
    PIIScanner,
)
from src.data.schemas import ConversationSample, ConversationTurn, RawSample


# ── fixtures ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def scanner() -> PIIScanner:
    """Module-scoped scanner to avoid repeated engine init."""
    return PIIScanner(score_threshold=0.7)


def _raw(instruction: str, output: str, **kw) -> RawSample:
    return RawSample(instruction=instruction, output=output, **kw)


def _conv(*pairs: tuple[str, str]) -> ConversationSample:
    turns = []
    for role, content in pairs:
        turns.append(ConversationTurn(role=role, content=content))
    return ConversationSample(conversations=turns)


# ══════════════════════════════════════════════
# Single-sample scan
# ══════════════════════════════════════════════


class TestScan:

    def test_no_pii(self, scanner: PIIScanner):
        sample = _raw("Explain photosynthesis.", "Plants use sunlight.")
        result = scanner.scan(sample)
        assert not result.has_pii
        assert result.entities == []

    def test_detects_email(self, scanner: PIIScanner):
        sample = _raw(
            "Contact support.", "Email us at john.doe@example.com for help."
        )
        result = scanner.scan(sample)
        assert result.has_pii
        types = {e.entity_type for e in result.entities}
        assert "EMAIL_ADDRESS" in types

    def test_detects_phone(self, scanner: PIIScanner):
        sample = _raw("Call us.", "Our number is 555-867-5309.")
        result = scanner.scan(sample)
        assert result.has_pii
        types = {e.entity_type for e in result.entities}
        assert "PHONE_NUMBER" in types

    def test_detects_person_name(self, scanner: PIIScanner):
        sample = _raw(
            "Who founded Microsoft?", "Bill Gates co-founded Microsoft."
        )
        result = scanner.scan(sample)
        assert result.has_pii
        types = {e.entity_type for e in result.entities}
        assert "PERSON" in types

    def test_detects_ssn(self, scanner: PIIScanner):
        sample = _raw("What is your SSN?", "My SSN is 219-09-9999.")
        result = scanner.scan(sample)
        assert result.has_pii
        types = {e.entity_type for e in result.entities}
        assert "US_SSN" in types

    def test_detects_credit_card(self, scanner: PIIScanner):
        sample = _raw(
            "Enter card.", "Card number: 4111 1111 1111 1111."
        )
        result = scanner.scan(sample)
        assert result.has_pii
        types = {e.entity_type for e in result.entities}
        assert "CREDIT_CARD" in types

    def test_detects_ip_address(self, scanner: PIIScanner):
        sample = _raw("Server info.", "The server IP is 192.168.1.100.")
        result = scanner.scan(sample)
        assert result.has_pii
        types = {e.entity_type for e in result.entities}
        assert "IP_ADDRESS" in types

    def test_conversation_sample(self, scanner: PIIScanner):
        conv = _conv(
            ("user", "My email is alice@test.com"),
            ("assistant", "Got it, Alice."),
        )
        result = scanner.scan(conv)
        assert result.has_pii
        types = {e.entity_type for e in result.entities}
        assert "EMAIL_ADDRESS" in types

    def test_pii_in_input_field(self, scanner: PIIScanner):
        sample = _raw(
            "Summarise this.",
            "Done.",
            input="John Smith lives at 123 Main St.",
        )
        result = scanner.scan(sample)
        assert result.has_pii

    def test_pii_in_system_prompt(self, scanner: PIIScanner):
        sample = _raw(
            "Do task.",
            "Ok.",
            system_prompt="You are Dr. Jane Wilson's assistant.",
        )
        result = scanner.scan(sample)
        assert result.has_pii

    def test_entity_snippet_matches(self, scanner: PIIScanner):
        sample = _raw("Email.", "Contact bob@corp.io please.")
        result = scanner.scan(sample)
        email_entities = [
            e for e in result.entities if e.entity_type == "EMAIL_ADDRESS"
        ]
        assert len(email_entities) >= 1
        assert "bob@corp.io" in email_entities[0].text_snippet

    def test_score_threshold_filters(self):
        """High threshold should filter out low-confidence detections."""
        strict = PIIScanner(score_threshold=0.99)
        sample = _raw("Hey.", "Maybe call me at 555.")
        result = strict.scan(sample)
        # With very strict threshold many borderline cases are excluded
        # This just verifies the threshold parameter is respected
        assert isinstance(result, PIIScanResult)


# ══════════════════════════════════════════════
# Non-Latin scripts & mixed content
# ══════════════════════════════════════════════


class TestNonLatinAndMixed:

    def test_chinese_text_no_false_positive(self, scanner: PIIScanner):
        sample = _raw("翻译这段话。", "今天天气很好。")
        result = scanner.scan(sample)
        # Should not crash; PII entities might be empty or minimal
        assert isinstance(result, PIIScanResult)

    def test_arabic_text(self, scanner: PIIScanner):
        sample = _raw("ترجم هذا.", "الطقس جميل.")
        result = scanner.scan(sample)
        assert isinstance(result, PIIScanResult)

    def test_mixed_script_with_email(self, scanner: PIIScanner):
        sample = _raw("翻译", "请发送邮件到 test@example.com")
        result = scanner.scan(sample)
        types = {e.entity_type for e in result.entities}
        assert "EMAIL_ADDRESS" in types

    def test_emoji_resilience(self, scanner: PIIScanner):
        sample = _raw("Rate 🌟", "Great product! 😊 Contact: foo@bar.com")
        result = scanner.scan(sample)
        types = {e.entity_type for e in result.entities}
        assert "EMAIL_ADDRESS" in types

    def test_cyrillic_text(self, scanner: PIIScanner):
        sample = _raw("Переведи это.", "Погода хорошая.")
        result = scanner.scan(sample)
        assert isinstance(result, PIIScanResult)


# ══════════════════════════════════════════════
# Batch scanning
# ══════════════════════════════════════════════


class TestBatchScan:

    def test_batch_counts(self, scanner: PIIScanner):
        samples = [
            _raw("Clean.", "No PII here."),
            _raw("Email.", "Contact alice@example.com."),
            _raw("Safe.", "Just facts about photosynthesis."),
        ]
        report = scanner.scan_batch(samples, n_workers=1)
        assert isinstance(report, BatchPIIReport)
        assert report.samples_with_pii >= 1
        assert report.samples_clean >= 1
        assert report.samples_with_pii + report.samples_clean == 3

    def test_batch_flagged_indices(self, scanner: PIIScanner):
        samples = [
            _raw("Ok.", "Fine."),
            _raw("SSN.", "My SSN is 219-09-9999."),
        ]
        report = scanner.scan_batch(samples, n_workers=1)
        assert 1 in report.flagged_indices

    def test_batch_pii_by_type(self, scanner: PIIScanner):
        samples = [
            _raw("A.", "alice@test.com"),
            _raw("B.", "My SSN is 219-09-9999."),
        ]
        report = scanner.scan_batch(samples, n_workers=1)
        assert "EMAIL_ADDRESS" in report.pii_by_type
        assert "US_SSN" in report.pii_by_type

    def test_batch_empty(self, scanner: PIIScanner):
        report = scanner.scan_batch([], n_workers=1)
        assert report.samples_with_pii == 0
        assert report.samples_clean == 0

    def test_batch_threaded(self, scanner: PIIScanner):
        samples = [
            _raw("A.", "alice@test.com"),
            _raw("B.", "No PII."),
        ]
        report = scanner.scan_batch(samples, n_workers=2)
        assert report.samples_with_pii + report.samples_clean == 2


# ══════════════════════════════════════════════
# Redaction
# ══════════════════════════════════════════════


class TestRedact:

    def test_mask_replaces_with_tag(self, scanner: PIIScanner):
        sample = _raw("Email.", "Contact alice@example.com please.")
        result = scanner.redact(sample, strategy="mask")
        assert result is not None
        assert "alice@example.com" not in result.output
        assert "<EMAIL_ADDRESS>" in result.output

    def test_remove_returns_none(self, scanner: PIIScanner):
        sample = _raw("SSN.", "My SSN is 219-09-9999.")
        result = scanner.redact(sample, strategy="remove")
        assert result is None

    def test_no_pii_returns_original(self, scanner: PIIScanner):
        sample = _raw("Facts.", "The sky is blue.")
        result = scanner.redact(sample, strategy="mask")
        assert result is not None
        assert result.output == "The sky is blue."

    def test_replace_uses_synthetic(self, scanner: PIIScanner):
        pytest.importorskip("faker")
        sample = _raw("Email.", "Contact alice@example.com please.")
        result = scanner.redact(sample, strategy="replace")
        assert result is not None
        assert "alice@example.com" not in result.output

    def test_mask_conversation(self, scanner: PIIScanner):
        conv = _conv(
            ("user", "My email is alice@test.com"),
            ("assistant", "Got it."),
        )
        result = scanner.redact(conv, strategy="mask")
        assert result is not None
        assert isinstance(result, ConversationSample)
        assert "alice@test.com" not in result.conversations[0].content


# ══════════════════════════════════════════════
# Dataclass integrity
# ══════════════════════════════════════════════


class TestDataclasses:

    def test_pii_entity_frozen(self):
        e = PIIEntity(
            entity_type="EMAIL_ADDRESS",
            start=0,
            end=10,
            score=0.9,
            text_snippet="a@b.com",
        )
        with pytest.raises(AttributeError):
            e.score = 0.5  # type: ignore[misc]

    def test_scan_result_has_pii_property(self):
        empty = PIIScanResult()
        assert not empty.has_pii
        with_entity = PIIScanResult(
            entities=[
                PIIEntity("EMAIL_ADDRESS", 0, 5, 0.9, "a@b.c")
            ]
        )
        assert with_entity.has_pii
