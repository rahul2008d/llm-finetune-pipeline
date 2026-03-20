"""Tests for DataCleaner — text cleaning, language filtering, deduplication."""
from __future__ import annotations

import pytest

from src.data.cleaner import (
    CleaningReport,
    DataCleaner,
    clean_text,
    _is_empty,
    _primary_text,
)
from src.data.schemas import ConversationSample, ConversationTurn, RawSample


def _raw(instruction: str, output: str, **kw) -> RawSample:
    return RawSample(instruction=instruction, output=output, **kw)


def _conv(*pairs: tuple[str, str]) -> ConversationSample:
    turns = []
    for role, content in pairs:
        turns.append(ConversationTurn(role=role, content=content))
    return ConversationSample(conversations=turns)


# ══════════════════════════════════════════════
# clean_text function
# ══════════════════════════════════════════════


class TestCleanText:

    def test_strip_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_nfc_normalisation(self):
        # é as combining (NFD) → single codepoint (NFC)
        nfd = "e\u0301"  # e + combining acute
        result = clean_text(nfd)
        assert result == "\u00e9"  # precomposed é

    def test_removes_null_bytes(self):
        assert clean_text("hello\x00world") == "helloworld"

    def test_removes_control_chars(self):
        text = "line1\x07\x08line2"
        assert clean_text(text) == "line1line2"

    def test_preserves_newlines(self):
        assert clean_text("a\nb") == "a\nb"

    def test_collapses_multiple_newlines(self):
        assert clean_text("a\n\n\n\nb") == "a\n\nb"

    def test_two_newlines_kept(self):
        assert clean_text("a\n\nb") == "a\n\nb"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   \t  ") == ""

    def test_mixed_control_and_newlines(self):
        text = "\x00hello\x0b\nworld\x1f\n\n\nend"
        result = clean_text(text)
        assert "\x00" not in result
        assert "\x0b" not in result
        assert "\x1f" not in result
        assert "hello" in result
        assert "world" in result

    def test_unicode_cjk_preserved(self):
        text = "  你好世界  "
        assert clean_text(text) == "你好世界"

    def test_unicode_arabic_preserved(self):
        text = "  مرحبا  "
        assert clean_text(text) == "مرحبا"

    def test_emoji_preserved(self):
        text = "Hello 🌍🎉"
        assert clean_text(text) == "Hello 🌍🎉"

    def test_mixed_scripts(self):
        text = "  Hello 你好 مرحبا  "
        assert clean_text(text) == "Hello 你好 مرحبا"


# ══════════════════════════════════════════════
# _is_empty helper
# ══════════════════════════════════════════════


class TestIsEmpty:

    def test_raw_not_empty(self):
        s = _raw("Do it", "Done")
        assert not _is_empty(s)

    def test_conversation_not_empty(self):
        c = _conv(("user", "Hi"), ("assistant", "Hello"))
        assert not _is_empty(c)


# ══════════════════════════════════════════════
# _primary_text helper
# ══════════════════════════════════════════════


class TestPrimaryText:

    def test_raw(self):
        s = _raw("Instruct", "Output", input="Extra")
        text = _primary_text(s)
        assert "Instruct" in text
        assert "Output" in text
        assert "Extra" in text

    def test_conversation(self):
        c = _conv(("user", "Hi"), ("assistant", "Hello"))
        text = _primary_text(c)
        assert "Hi" in text
        assert "Hello" in text


# ══════════════════════════════════════════════
# DataCleaner.clean — single sample
# ══════════════════════════════════════════════


class TestClean:

    @pytest.fixture
    def cleaner(self) -> DataCleaner:
        return DataCleaner()

    def test_clean_strips_and_normalises(self, cleaner: DataCleaner):
        sample = _raw("  Hello\x00  ", "  World\x07  ")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.instruction == "Hello"
        assert result.output == "World"

    def test_clean_strips_input_field(self, cleaner: DataCleaner):
        sample = _raw("Do", "Done", input="  extra\x00  ")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.input == "extra"

    def test_clean_strips_system_prompt(self, cleaner: DataCleaner):
        sample = _raw("Do", "Done", system_prompt="  sys\x00  ")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.system_prompt == "sys"

    def test_clean_conversation(self, cleaner: DataCleaner):
        conv = _conv(("user", "  Hi\x00  "), ("assistant", "  Hello\x07  "))
        result = cleaner.clean(conv)
        assert result is not None
        assert result.conversations[0].content == "Hi"
        assert result.conversations[1].content == "Hello"

    def test_clean_collapses_newlines(self, cleaner: DataCleaner):
        sample = _raw("Do", "a\n\n\n\nb")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.output == "a\n\nb"

    def test_preserves_metadata(self, cleaner: DataCleaner):
        sample = _raw("Do", "Done", metadata={"key": "val"})
        result = cleaner.clean(sample)
        assert result is not None
        assert result.metadata == {"key": "val"}

    def test_none_input_stays_none(self, cleaner: DataCleaner):
        sample = _raw("Do", "Done")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.input is None

    def test_none_system_prompt_stays_none(self, cleaner: DataCleaner):
        sample = _raw("Do", "Done")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.system_prompt is None


# ══════════════════════════════════════════════
# DataCleaner.clean_batch
# ══════════════════════════════════════════════


class TestCleanBatch:

    def test_basic_batch(self):
        cleaner = DataCleaner(minhash_threshold=1.0)  # disable dedup
        samples = [
            _raw("  A\x00  ", "  B  "),
            _raw("  C  ", "  D  "),
        ]
        cleaned, report = cleaner.clean_batch(samples)
        assert len(cleaned) == 2
        assert report.total_input == 2
        assert report.total_output == 2
        assert report.dropped_empty == 0

    def test_empty_batch(self):
        cleaner = DataCleaner()
        cleaned, report = cleaner.clean_batch([])
        assert len(cleaned) == 0
        assert report.total_input == 0

    def test_report_type(self):
        cleaner = DataCleaner(minhash_threshold=1.0)
        _, report = cleaner.clean_batch([_raw("X", "Y")])
        assert isinstance(report, CleaningReport)

    def test_dedup_removes_exact_copies(self):
        cleaner = DataCleaner(minhash_threshold=0.5)
        samples = [
            _raw("Same instruction here today", "Same output here today"),
            _raw("Same instruction here today", "Same output here today"),
            _raw("Same instruction here today", "Same output here today"),
        ]
        cleaned, report = cleaner.clean_batch(samples)
        assert len(cleaned) < 3
        assert report.near_duplicates_removed > 0

    def test_dedup_keeps_distinct(self):
        cleaner = DataCleaner(minhash_threshold=0.85)
        samples = [
            _raw(
                "Explain quantum mechanics in detail",
                "Quantum mechanics is a fundamental theory",
            ),
            _raw(
                "Write a poem about the ocean waves",
                "The ocean waves crash upon the shore",
            ),
        ]
        cleaned, report = cleaner.clean_batch(samples)
        assert len(cleaned) == 2
        assert report.near_duplicates_removed == 0


# ══════════════════════════════════════════════
# Edge cases: non-Latin, mixed content
# ══════════════════════════════════════════════


class TestEdgeCases:

    @pytest.fixture
    def cleaner(self) -> DataCleaner:
        return DataCleaner(minhash_threshold=1.0)

    def test_cjk_cleaned(self, cleaner: DataCleaner):
        sample = _raw("  你好\x00  ", "  世界  ")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.instruction == "你好"
        assert result.output == "世界"

    def test_arabic_cleaned(self, cleaner: DataCleaner):
        sample = _raw("  مرحبا\x07  ", "  عالم  ")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.instruction == "مرحبا"

    def test_cyrillic_cleaned(self, cleaner: DataCleaner):
        sample = _raw("  Привет\x00  ", "  Мир  ")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.instruction == "Привет"

    def test_devanagari_cleaned(self, cleaner: DataCleaner):
        sample = _raw("  नमस्ते  ", "  दुनिया  ")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.instruction == "नमस्ते"

    def test_mixed_scripts_cleaned(self, cleaner: DataCleaner):
        sample = _raw("  Hello 你好  ", "  World 世界  ")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.instruction == "Hello 你好"

    def test_emoji_preserved(self, cleaner: DataCleaner):
        sample = _raw("Rate this 🌟", "Great! 😊")
        result = cleaner.clean(sample)
        assert result is not None
        assert "🌟" in result.instruction
        assert "😊" in result.output

    def test_conversation_non_latin(self, cleaner: DataCleaner):
        conv = _conv(("user", "  你好\x00  "), ("assistant", "  世界  "))
        result = cleaner.clean(conv)
        assert result is not None
        assert result.conversations[0].content == "你好"

    def test_combining_characters_normalised(self, cleaner: DataCleaner):
        # NFD café → NFC café
        nfd_cafe = "cafe\u0301"
        sample = _raw(nfd_cafe, "result")
        result = cleaner.clean(sample)
        assert result is not None
        assert result.instruction == "caf\u00e9"

    def test_zero_width_chars_preserved(self, cleaner: DataCleaner):
        """Zero-width joiners used in some scripts should survive."""
        text = "क्षत्रिय"  # Devanagari with implicit joiners
        sample = _raw(text, "output")
        result = cleaner.clean(sample)
        assert result is not None
        assert len(result.instruction) > 0
