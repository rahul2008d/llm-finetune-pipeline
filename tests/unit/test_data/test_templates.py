"""Tests for TemplateEngine — template rendering, special tokens, edge cases."""
from __future__ import annotations

import pytest

from src.data.schemas import ConversationSample, ConversationTurn, RawSample
from src.data.templates import TemplateEngine


# ── helpers ─────────────────────────────────────────────────────


def _raw(
    instruction: str = "Summarise this.",
    output: str = "Done.",
    *,
    input: str | None = None,
    system_prompt: str | None = None,
) -> RawSample:
    return RawSample(
        instruction=instruction,
        output=output,
        input=input,
        system_prompt=system_prompt,
    )


def _conv(*pairs: tuple[str, str]) -> ConversationSample:
    turns = [ConversationTurn(role=r, content=c) for r, c in pairs]
    return ConversationSample(conversations=turns)


# ══════════════════════════════════════════════
# Alpaca template
# ══════════════════════════════════════════════


class TestAlpacaTemplate:

    @pytest.fixture
    def engine(self) -> TemplateEngine:
        return TemplateEngine("alpaca")

    def test_basic_render(self, engine: TemplateEngine):
        result = engine.apply(_raw("Explain X.", "X is Y."))
        assert "### Instruction:" in result
        assert "Explain X." in result
        assert "### Response:" in result
        assert "X is Y." in result

    def test_with_input(self, engine: TemplateEngine):
        result = engine.apply(
            _raw("Translate", "Hola", input="Hello")
        )
        assert "### Input:" in result
        assert "Hello" in result

    def test_without_input_no_input_section(self, engine: TemplateEngine):
        result = engine.apply(_raw("Do it", "Done"))
        assert "### Input:" not in result

    def test_response_start_token(self, engine: TemplateEngine):
        assert engine.get_response_start_token() == "### Response:\n"

    def test_special_tokens_present(self, engine: TemplateEngine):
        result = engine.apply(_raw())
        assert "### Instruction:" in result
        assert "### Response:" in result

    def test_missing_system_prompt_graceful(self, engine: TemplateEngine):
        result = engine.apply(_raw("Do", "Done"))
        # Should render without error even though no system_prompt
        assert "Do" in result
        assert "Done" in result


# ══════════════════════════════════════════════
# ChatML template
# ══════════════════════════════════════════════


class TestChatMLTemplate:

    @pytest.fixture
    def engine(self) -> TemplateEngine:
        return TemplateEngine("chatml")

    def test_basic_render(self, engine: TemplateEngine):
        result = engine.apply(_raw("Hello", "Hi"))
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "<|im_end|>" in result
        assert "Hello" in result
        assert "Hi" in result

    def test_with_system(self, engine: TemplateEngine):
        result = engine.apply(
            _raw("Hello", "Hi", system_prompt="Be helpful")
        )
        assert "<|im_start|>system" in result
        assert "Be helpful" in result

    def test_without_system(self, engine: TemplateEngine):
        result = engine.apply(_raw("Hello", "Hi"))
        assert "<|im_start|>system" not in result

    def test_response_start_token(self, engine: TemplateEngine):
        assert engine.get_response_start_token() == "<|im_start|>assistant\n"

    def test_special_tokens_present(self, engine: TemplateEngine):
        result = engine.apply(_raw())
        assert "<|im_start|>" in result
        assert "<|im_end|>" in result

    def test_im_end_count(self, engine: TemplateEngine):
        result = engine.apply(_raw("Q", "A"))
        # Without system: user + assistant = 2 im_end
        assert result.count("<|im_end|>") == 2

    def test_im_end_count_with_system(self, engine: TemplateEngine):
        result = engine.apply(_raw("Q", "A", system_prompt="S"))
        # system + user + assistant = 3 im_end
        assert result.count("<|im_end|>") == 3


# ══════════════════════════════════════════════
# Llama 3 template
# ══════════════════════════════════════════════


class TestLlama3Template:

    @pytest.fixture
    def engine(self) -> TemplateEngine:
        return TemplateEngine("llama3")

    def test_basic_render(self, engine: TemplateEngine):
        result = engine.apply(_raw("Hello", "Hi"))
        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>user<|end_header_id|>" in result
        assert "<|start_header_id|>assistant<|end_header_id|>" in result
        assert "<|eot_id|>" in result

    def test_with_system(self, engine: TemplateEngine):
        result = engine.apply(
            _raw("Q", "A", system_prompt="System msg")
        )
        assert "<|start_header_id|>system<|end_header_id|>" in result
        assert "System msg" in result

    def test_without_system(self, engine: TemplateEngine):
        result = engine.apply(_raw("Q", "A"))
        assert "<|start_header_id|>system<|end_header_id|>" not in result

    def test_response_start_token(self, engine: TemplateEngine):
        token = engine.get_response_start_token()
        assert token == "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def test_begin_of_text(self, engine: TemplateEngine):
        result = engine.apply(_raw())
        assert result.startswith("<|begin_of_text|>")

    def test_ends_with_eot(self, engine: TemplateEngine):
        result = engine.apply(_raw())
        assert result.rstrip().endswith("<|eot_id|>")

    def test_eot_count_no_system(self, engine: TemplateEngine):
        result = engine.apply(_raw("Q", "A"))
        # user + assistant = 2 eot_id
        assert result.count("<|eot_id|>") == 2

    def test_eot_count_with_system(self, engine: TemplateEngine):
        result = engine.apply(_raw("Q", "A", system_prompt="S"))
        # system + user + assistant = 3 eot_id
        assert result.count("<|eot_id|>") == 3


# ══════════════════════════════════════════════
# Mistral template
# ══════════════════════════════════════════════


class TestMistralTemplate:

    @pytest.fixture
    def engine(self) -> TemplateEngine:
        return TemplateEngine("mistral")

    def test_basic_render(self, engine: TemplateEngine):
        result = engine.apply(_raw("Hello", "Hi"))
        assert "[INST]" in result
        assert "[/INST]" in result
        assert "</s>" in result

    def test_with_system(self, engine: TemplateEngine):
        result = engine.apply(
            _raw("Q", "A", system_prompt="Be concise")
        )
        assert "Be concise" in result
        assert "[INST]" in result

    def test_response_start_token(self, engine: TemplateEngine):
        assert engine.get_response_start_token() == "[/INST] "

    def test_ends_with_eos(self, engine: TemplateEngine):
        result = engine.apply(_raw())
        assert result.rstrip().endswith("</s>")

    def test_special_tokens_present(self, engine: TemplateEngine):
        result = engine.apply(_raw())
        assert "[INST]" in result
        assert "[/INST]" in result
        assert "</s>" in result


# ══════════════════════════════════════════════
# Multi-turn conversations
# ══════════════════════════════════════════════


class TestMultiTurnConversation:

    def test_chatml_multi_turn(self):
        engine = TemplateEngine("chatml")
        conv = _conv(
            ("user", "Hi"),
            ("assistant", "Hello!"),
            ("user", "How are you?"),
            ("assistant", "I'm fine!"),
        )
        result = engine.apply(conv)
        assert "Hi" in result
        assert "Hello!" in result
        assert "How are you?" in result
        assert "I'm fine!" in result

    def test_chatml_conversation_with_system(self):
        engine = TemplateEngine("chatml")
        conv = _conv(
            ("system", "Be helpful"),
            ("user", "Hi"),
            ("assistant", "Hello!"),
        )
        result = engine.apply(conv)
        assert "<|im_start|>system" in result
        assert "Be helpful" in result

    def test_llama3_multi_turn(self):
        engine = TemplateEngine("llama3")
        conv = _conv(
            ("user", "First question"),
            ("assistant", "First answer"),
            ("user", "Second question"),
            ("assistant", "Second answer"),
        )
        result = engine.apply(conv)
        assert "First question" in result
        assert "Second answer" in result

    def test_alpaca_multi_turn(self):
        engine = TemplateEngine("alpaca")
        conv = _conv(
            ("user", "Turn 1"),
            ("assistant", "Reply 1"),
        )
        result = engine.apply(conv)
        assert "Turn 1" in result
        assert "Reply 1" in result

    def test_system_only_in_first_turn(self):
        engine = TemplateEngine("chatml")
        conv = _conv(
            ("system", "System instruction"),
            ("user", "Q1"),
            ("assistant", "A1"),
            ("user", "Q2"),
            ("assistant", "A2"),
        )
        result = engine.apply(conv)
        # System should appear only once (in the first rendered chunk)
        assert result.count("System instruction") == 1


# ══════════════════════════════════════════════
# Custom templates
# ══════════════════════════════════════════════


class TestCustomTemplate:

    def test_custom_template_string(self):
        engine = TemplateEngine(
            "custom",
            custom_template="Q: {{ instruction }}\nA: {{ output }}",
        )
        result = engine.apply(_raw("What?", "This."))
        assert result == "Q: What?\nA: This."

    def test_custom_template_from_path(self, tmp_path):
        p = tmp_path / "my.jinja2"
        p.write_text("USER: {{ instruction }}\nBOT: {{ output }}")
        engine = TemplateEngine(p)
        result = engine.apply(_raw("Hey", "Hi"))
        assert result == "USER: Hey\nBOT: Hi"

    def test_custom_response_start_token(self):
        engine = TemplateEngine(
            "custom",
            custom_template="{{ instruction }} -> {{ output }}",
        )
        # Unknown template name returns empty string
        assert engine.get_response_start_token() == ""


# ══════════════════════════════════════════════
# Batch apply
# ══════════════════════════════════════════════


class TestBatchApply:

    def test_batch_sequential(self):
        engine = TemplateEngine("alpaca")
        samples = [_raw("A", "B"), _raw("C", "D")]
        results = engine.apply_batch(samples, n_workers=1)
        assert len(results) == 2
        assert "A" in results[0]
        assert "C" in results[1]

    def test_batch_parallel(self):
        engine = TemplateEngine("alpaca")
        samples = [_raw("A", "B"), _raw("C", "D"), _raw("E", "F")]
        results = engine.apply_batch(samples, n_workers=2)
        assert len(results) == 3

    def test_batch_empty(self):
        engine = TemplateEngine("alpaca")
        assert engine.apply_batch([], n_workers=1) == []


# ══════════════════════════════════════════════
# Token estimation
# ══════════════════════════════════════════════


class TestTokenEstimation:

    def test_estimate_returns_positive(self):
        engine = TemplateEngine("alpaca")
        count = engine.estimate_tokens(_raw("A B C", "D E"))
        assert count > 0

    def test_longer_text_more_tokens(self):
        engine = TemplateEngine("alpaca")
        short = engine.estimate_tokens(_raw("A", "B"))
        long = engine.estimate_tokens(
            _raw("A B C D E F G H I J", "K L M N O P Q R S T")
        )
        assert long > short


# ══════════════════════════════════════════════
# Error handling
# ══════════════════════════════════════════════


class TestErrors:

    def test_nonexistent_template_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            TemplateEngine("nonexistent_template_xyz")

    def test_sandboxed_no_imports(self):
        """Jinja2 sandbox should prevent dangerous operations."""
        engine = TemplateEngine(
            "test",
            custom_template="{{ instruction.__class__.__mro__ }}",
        )
        # Should raise SecurityError from sandbox
        with pytest.raises(Exception):
            engine.apply(_raw("X", "Y"))


# ══════════════════════════════════════════════
# Missing fields handled gracefully
# ══════════════════════════════════════════════


class TestMissingFields:

    def test_no_input_alpaca(self):
        engine = TemplateEngine("alpaca")
        result = engine.apply(_raw("Do it", "Done"))
        # Should render cleanly without ### Input: section
        assert "### Input:" not in result

    def test_no_system_chatml(self):
        engine = TemplateEngine("chatml")
        result = engine.apply(_raw("Q", "A"))
        assert "<|im_start|>system" not in result

    def test_no_system_llama3(self):
        engine = TemplateEngine("llama3")
        result = engine.apply(_raw("Q", "A"))
        assert "<|start_header_id|>system<|end_header_id|>" not in result

    def test_no_system_mistral(self):
        engine = TemplateEngine("mistral")
        result = engine.apply(_raw("Q", "A"))
        # Without system, should still have [INST] and [/INST]
        assert "[INST]" in result
        assert "Q" in result
