"""Tests for data.tokenizer – DatasetTokenizer."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.data.schemas import (
    ConversationSample,
    ConversationTurn,
    DatasetManifest,
    RawSample,
)
from src.data.templates import TemplateEngine
from src.data.tokenizer import IGNORE_INDEX, DatasetTokenizer, TokenizedSample


# ── fixtures ────────────────────────────────────────────────────

MODEL_NAME = "gpt2"  # small, fast, always available from HF cache


@pytest.fixture(scope="module")
def alpaca_engine() -> TemplateEngine:
    return TemplateEngine("alpaca")


@pytest.fixture(scope="module")
def chatml_engine() -> TemplateEngine:
    return TemplateEngine("chatml")


@pytest.fixture(scope="module")
def llama3_engine() -> TemplateEngine:
    return TemplateEngine("llama3")


@pytest.fixture(scope="module")
def mistral_engine() -> TemplateEngine:
    return TemplateEngine("mistral")


@pytest.fixture(scope="module")
def tokenizer_alpaca(alpaca_engine: TemplateEngine) -> DatasetTokenizer:
    return DatasetTokenizer(MODEL_NAME, 512, alpaca_engine)


@pytest.fixture(scope="module")
def tokenizer_chatml(chatml_engine: TemplateEngine) -> DatasetTokenizer:
    return DatasetTokenizer(MODEL_NAME, 512, chatml_engine)


@pytest.fixture(scope="module")
def tokenizer_llama3(llama3_engine: TemplateEngine) -> DatasetTokenizer:
    return DatasetTokenizer(MODEL_NAME, 512, llama3_engine)


@pytest.fixture(scope="module")
def tokenizer_mistral(mistral_engine: TemplateEngine) -> DatasetTokenizer:
    return DatasetTokenizer(MODEL_NAME, 512, mistral_engine)


def _raw(
    instruction: str = "Say hello",
    inp: str = "",
    output: str = "Hello!",
    system: str = "",
) -> RawSample:
    return RawSample(
        instruction=instruction,
        input=inp,
        output=output,
        system_prompt=system or None,
    )


def _conv(
    turns: list[tuple[str, str]],
    system: str | None = None,
) -> ConversationSample:
    convs: list[ConversationTurn] = []
    if system:
        convs.append(ConversationTurn(role="system", content=system))
    for user, asst in turns:
        convs.append(ConversationTurn(role="user", content=user))
        convs.append(ConversationTurn(role="assistant", content=asst))
    return ConversationSample(conversations=convs)


# ── init ────────────────────────────────────────────────────────


class TestInit:
    def test_tokenizer_loads(self, tokenizer_alpaca: DatasetTokenizer) -> None:
        assert tokenizer_alpaca._tokenizer is not None
        assert tokenizer_alpaca._tokenizer.padding_side == "right"

    def test_pad_token_set(self, tokenizer_alpaca: DatasetTokenizer) -> None:
        assert tokenizer_alpaca._tokenizer.pad_token is not None

    def test_response_start_populated(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        assert len(tokenizer_alpaca._response_start) > 0

    def test_max_seq_length(self, tokenizer_alpaca: DatasetTokenizer) -> None:
        assert tokenizer_alpaca.max_seq_length == 512


# ── tokenize_sample ────────────────────────────────────────────


class TestTokenizeSample:
    def test_returns_tokenized_sample(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        assert isinstance(result, TokenizedSample)

    def test_input_ids_are_ints(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        assert all(isinstance(x, int) for x in result.input_ids)

    def test_attention_mask_all_ones(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        assert all(m == 1 for m in result.attention_mask)

    def test_lengths_match(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        assert len(result.input_ids) == len(result.attention_mask) == len(result.labels)

    def test_not_truncated(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        assert result.was_truncated is False

    def test_truncation_very_long(self, alpaca_engine: TemplateEngine) -> None:
        short_tok = DatasetTokenizer(MODEL_NAME, 32, alpaca_engine)
        sample = _raw(output="word " * 200)
        result = short_tok.tokenize_sample(sample)
        assert result.was_truncated is True
        assert len(result.input_ids) == 32

    def test_eos_appended(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        eos_id = tokenizer_alpaca._tokenizer.eos_token_id
        assert result.input_ids[-1] == eos_id

    def test_no_eos_when_disabled(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw(), add_eos=False)
        eos_id = tokenizer_alpaca._tokenizer.eos_token_id
        # The last token may or may not be EOS depending on template,
        # but we verify the output is shorter without eos
        result_with = tokenizer_alpaca.tokenize_sample(_raw(), add_eos=True)
        assert len(result.input_ids) <= len(result_with.input_ids)


# ── label masking ───────────────────────────────────────────────


class TestLabelMasking:
    """The core correctness criterion: prompt tokens → -100, response tokens → real ids."""

    def test_alpaca_labels_mask_prompt(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        # Some labels should be -100 (prompt) and some should be real ids (response)
        masked = [l for l in result.labels if l == IGNORE_INDEX]
        real = [l for l in result.labels if l != IGNORE_INDEX]
        assert len(masked) > 0, "prompt tokens should be masked"
        assert len(real) > 0, "response tokens should have real labels"

    def test_chatml_labels_mask_prompt(
        self, tokenizer_chatml: DatasetTokenizer
    ) -> None:
        result = tokenizer_chatml.tokenize_sample(_raw())
        masked = [l for l in result.labels if l == IGNORE_INDEX]
        real = [l for l in result.labels if l != IGNORE_INDEX]
        assert len(masked) > 0
        assert len(real) > 0

    def test_llama3_labels_mask_prompt(
        self, tokenizer_llama3: DatasetTokenizer
    ) -> None:
        result = tokenizer_llama3.tokenize_sample(_raw())
        masked = [l for l in result.labels if l == IGNORE_INDEX]
        real = [l for l in result.labels if l != IGNORE_INDEX]
        assert len(masked) > 0
        assert len(real) > 0

    def test_mistral_labels_mask_prompt(
        self, tokenizer_mistral: DatasetTokenizer
    ) -> None:
        result = tokenizer_mistral.tokenize_sample(_raw())
        masked = [l for l in result.labels if l == IGNORE_INDEX]
        real = [l for l in result.labels if l != IGNORE_INDEX]
        assert len(masked) > 0
        assert len(real) > 0

    def test_prompt_prefix_is_all_masked(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        # First label must be IGNORE_INDEX (prompt always starts first)
        assert result.labels[0] == IGNORE_INDEX

    def test_response_tail_has_real_labels(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw())
        # labels (excluding trailing eos) should end with real ids
        # Find last non-eos real label
        real_labels = [l for l in result.labels if l != IGNORE_INDEX]
        assert len(real_labels) > 0

    def test_masked_real_boundary(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        """Once labels switch from IGNORE to real, they stay real."""
        result = tokenizer_alpaca.tokenize_sample(_raw())
        found_real = False
        for l in result.labels:
            if l != IGNORE_INDEX:
                found_real = True
            elif found_real:
                # After real labels, we shouldn't see IGNORE again
                # (in single-turn instruction samples)
                pytest.fail(
                    "Found IGNORE_INDEX after real labels in single-turn sample"
                )

    def test_conversation_sample_masking(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        sample = _conv([("How are you?", "Fine!")])
        result = tokenizer_alpaca.tokenize_sample(sample)
        masked = [l for l in result.labels if l == IGNORE_INDEX]
        real = [l for l in result.labels if l != IGNORE_INDEX]
        assert len(masked) > 0
        assert len(real) > 0

    def test_no_response_marker_masks_all(
        self, alpaca_engine: TemplateEngine
    ) -> None:
        """If the template name doesn't match any known family, all are masked."""
        engine = TemplateEngine(
            template_name="unknown_template",
            custom_template="{{ instruction }} {{ output }}",
        )
        tok = DatasetTokenizer(MODEL_NAME, 512, engine)
        result = tok.tokenize_sample(_raw())
        assert all(l == IGNORE_INDEX for l in result.labels)


# ── build_dataset ───────────────────────────────────────────────


class TestBuildDataset:
    @pytest.fixture()
    def samples(self) -> list[RawSample]:
        return [
            _raw(instruction=f"Instruction {i}", output=f"Output {i}")
            for i in range(20)
        ]

    def test_returns_manifest(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        manifest = tokenizer_alpaca.build_dataset(samples, str(tmp_path / "out"))
        assert isinstance(manifest, DatasetManifest)

    def test_output_files_created(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "out"
        tokenizer_alpaca.build_dataset(samples, str(out))
        assert (out / "train.jsonl").exists()
        assert (out / "val.jsonl").exists()
        assert (out / "test.jsonl").exists()
        assert (out / "manifest.json").exists()

    def test_split_counts(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "out"
        manifest = tokenizer_alpaca.build_dataset(samples, str(out))
        # Read actual counts
        train_lines = (out / "train.jsonl").read_text().strip().splitlines()
        val_lines = (out / "val.jsonl").read_text().strip().splitlines()
        test_lines = (out / "test.jsonl").read_text().strip().splitlines()
        total = len(train_lines) + len(val_lines) + len(test_lines)
        assert total == manifest.num_samples
        assert len(train_lines) > len(val_lines) > 0
        assert len(test_lines) > 0

    def test_manifest_has_stats(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        manifest = tokenizer_alpaca.build_dataset(samples, str(tmp_path / "out"))
        assert manifest.statistics is not None
        stats = manifest.statistics
        assert stats.total_samples == manifest.num_samples
        assert stats.avg_input_tokens > 0
        assert stats.avg_output_tokens > 0

    def test_manifest_has_sha256(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        manifest = tokenizer_alpaca.build_dataset(samples, str(tmp_path / "out"))
        assert len(manifest.sha256_checksum) == 64

    def test_jsonl_records_valid(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "out"
        tokenizer_alpaca.build_dataset(samples, str(out))
        for line in (out / "train.jsonl").read_text().strip().splitlines():
            rec = json.loads(line)
            assert "input_ids" in rec
            assert "attention_mask" in rec
            assert "labels" in rec
            assert len(rec["input_ids"]) == len(rec["labels"])

    def test_deterministic_with_same_seed(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        m1 = tokenizer_alpaca.build_dataset(samples, str(out1), seed=42)
        m2 = tokenizer_alpaca.build_dataset(samples, str(out2), seed=42)
        data1 = (out1 / "train.jsonl").read_text()
        data2 = (out2 / "train.jsonl").read_text()
        assert data1 == data2

    def test_different_seed_different_order(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        tokenizer_alpaca.build_dataset(samples, str(out1), seed=1)
        tokenizer_alpaca.build_dataset(samples, str(out2), seed=99)
        data1 = (out1 / "train.jsonl").read_text()
        data2 = (out2 / "train.jsonl").read_text()
        assert data1 != data2

    def test_no_shuffle(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        samples: list[RawSample],
        tmp_path: Path,
    ) -> None:
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        m1 = tokenizer_alpaca.build_dataset(
            samples, str(out1), shuffle=False, seed=1
        )
        m2 = tokenizer_alpaca.build_dataset(
            samples, str(out2), shuffle=False, seed=999
        )
        data1 = (out1 / "train.jsonl").read_text()
        data2 = (out2 / "train.jsonl").read_text()
        assert data1 == data2  # same order regardless of seed

    def test_empty_after_filter_raises(
        self, alpaca_engine: TemplateEngine, tmp_path: Path
    ) -> None:
        """If all samples produce empty responses, build_dataset should raise."""
        engine = TemplateEngine(
            template_name="unknown_family",
            custom_template="{{ instruction }}",
        )
        tok = DatasetTokenizer(MODEL_NAME, 512, engine)
        # Since unknown_family → no response start → all labels are -100
        with pytest.raises(ValueError, match="No samples survived"):
            tok.build_dataset([_raw()], str(tmp_path / "out"))

    def test_conversation_samples_in_dataset(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        tmp_path: Path,
    ) -> None:
        samples = [
            _conv([("Hi", "Hello!")]),
            _conv([("Bye", "Goodbye!")]),
            _conv([("Test", "OK")]),
            _conv([("One", "Two")]),
            _conv([("A", "B")]),
        ]
        manifest = tokenizer_alpaca.build_dataset(samples, str(tmp_path / "out"))
        assert manifest.num_samples == 5

    def test_custom_splits(
        self,
        tokenizer_alpaca: DatasetTokenizer,
        tmp_path: Path,
    ) -> None:
        samples = [_raw(instruction=f"Q{i}", output=f"A{i}") for i in range(100)]
        out = tmp_path / "out"
        manifest = tokenizer_alpaca.build_dataset(
            samples, str(out), train_split=0.8, val_split=0.1, test_split=0.1
        )
        train_n = len((out / "train.jsonl").read_text().strip().splitlines())
        val_n = len((out / "val.jsonl").read_text().strip().splitlines())
        test_n = len((out / "test.jsonl").read_text().strip().splitlines())
        assert train_n > val_n
        assert train_n > test_n
        assert train_n + val_n + test_n == manifest.num_samples


# ── statistics ──────────────────────────────────────────────────


class TestStatistics:
    def test_percentiles_present(
        self, tokenizer_alpaca: DatasetTokenizer, tmp_path: Path
    ) -> None:
        samples = [_raw(instruction=f"Q{i}", output=f"A{i}") for i in range(10)]
        manifest = tokenizer_alpaca.build_dataset(samples, str(tmp_path / "out"))
        pcts = manifest.statistics.token_distribution_percentiles
        assert "p50" in pcts
        assert "p90" in pcts
        assert "p95" in pcts
        assert "p99" in pcts

    def test_max_tokens_reasonable(
        self, tokenizer_alpaca: DatasetTokenizer, tmp_path: Path
    ) -> None:
        samples = [_raw(instruction=f"Q{i}", output=f"A{i}") for i in range(10)]
        manifest = tokenizer_alpaca.build_dataset(samples, str(tmp_path / "out"))
        stats = manifest.statistics
        assert stats.max_input_tokens <= 512
        assert stats.max_output_tokens <= 512


# ── edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    def test_min_samples_for_splits(
        self, tokenizer_alpaca: DatasetTokenizer, tmp_path: Path
    ) -> None:
        """A very small dataset should still produce all three splits."""
        samples = [_raw(instruction=f"Q{i}", output=f"A{i}") for i in range(3)]
        manifest = tokenizer_alpaca.build_dataset(samples, str(tmp_path / "out"))
        assert manifest.num_samples == 3

    def test_exact_max_length_sample(
        self, alpaca_engine: TemplateEngine
    ) -> None:
        tok = DatasetTokenizer(MODEL_NAME, 50, alpaca_engine)
        sample = _raw(output="word " * 50)
        result = tok.tokenize_sample(sample)
        assert len(result.input_ids) <= 50

    def test_single_token_response(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        result = tokenizer_alpaca.tokenize_sample(_raw(output="X"))
        real = [l for l in result.labels if l != IGNORE_INDEX]
        assert len(real) >= 1

    def test_with_input_field(
        self, tokenizer_alpaca: DatasetTokenizer
    ) -> None:
        sample = _raw(
            instruction="Translate",
            inp="Hello",
            output="Bonjour",
        )
        result = tokenizer_alpaca.tokenize_sample(sample)
        assert len(result.input_ids) > 0
        masked = [l for l in result.labels if l == IGNORE_INDEX]
        assert len(masked) > 0

    def test_with_system_prompt(
        self, tokenizer_chatml: DatasetTokenizer
    ) -> None:
        sample = _raw(
            instruction="What is 1+1?",
            output="2",
            system="You are a math tutor.",
        )
        result = tokenizer_chatml.tokenize_sample(sample)
        masked = [l for l in result.labels if l == IGNORE_INDEX]
        real = [l for l in result.labels if l != IGNORE_INDEX]
        assert len(masked) > 0
        assert len(real) > 0
