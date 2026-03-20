"""Tests for format converters — unit tests + Hypothesis property-based tests.

Converter round-trip invariants tested via Hypothesis:
  - instruction → conversation → instruction preserves key fields
  - alpaca dict → conversation preserves content
  - sharegpt dict → conversation preserves content
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pydantic import ValidationError

from src.data.converters import (
    alpaca_to_conversation,
    conversation_to_instruction,
    instruction_to_conversation,
    sharegpt_to_conversation,
)
from src.data.schemas import ConversationSample, ConversationTurn, RawSample


# ══════════════════════════════════════════════
# Strategies for Hypothesis
# ══════════════════════════════════════════════

# Non-empty, reasonably sized text for fields
_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip())

_optional_text = st.one_of(st.none(), _text)


@st.composite
def raw_samples(draw):
    """Generate valid RawSample instances."""
    instruction = draw(_text)
    output = draw(_text)
    inp = draw(_optional_text)
    sys_prompt = draw(_optional_text)
    return RawSample(
        instruction=instruction,
        output=output,
        input=inp,
        system_prompt=sys_prompt,
    )


@st.composite
def simple_conversations(draw):
    """Generate valid 2-turn ConversationSample (user → assistant)."""
    user_content = draw(_text)
    assistant_content = draw(_text)
    return ConversationSample(
        conversations=[
            ConversationTurn(role="user", content=user_content),
            ConversationTurn(role="assistant", content=assistant_content),
        ]
    )


@st.composite
def conversations_with_system(draw):
    """Generate valid 3-turn ConversationSample (system → user → assistant)."""
    sys_content = draw(_text)
    user_content = draw(_text)
    assistant_content = draw(_text)
    return ConversationSample(
        conversations=[
            ConversationTurn(role="system", content=sys_content),
            ConversationTurn(role="user", content=user_content),
            ConversationTurn(role="assistant", content=assistant_content),
        ]
    )


# ══════════════════════════════════════════════
# instruction_to_conversation — unit tests
# ══════════════════════════════════════════════


class TestInstructionToConversation:

    def test_minimal(self):
        sample = RawSample(instruction="Explain X", output="X is ...")
        conv = instruction_to_conversation(sample)
        assert isinstance(conv, ConversationSample)
        assert len(conv.conversations) == 2
        assert conv.conversations[0].role == "user"
        assert conv.conversations[0].content == "Explain X"
        assert conv.conversations[1].role == "assistant"
        assert conv.conversations[1].content == "X is ..."

    def test_with_system_prompt(self):
        sample = RawSample(
            instruction="Translate",
            output="Hola",
            system_prompt="You are a translator.",
        )
        conv = instruction_to_conversation(sample)
        assert len(conv.conversations) == 3
        assert conv.conversations[0].role == "system"
        assert conv.conversations[0].content == "You are a translator."

    def test_with_input_appended(self):
        sample = RawSample(
            instruction="Summarise", input="Long text here", output="Short"
        )
        conv = instruction_to_conversation(sample)
        assert "Long text here" in conv.conversations[0].content
        assert "Summarise" in conv.conversations[0].content

    def test_no_side_effects(self):
        sample = RawSample(instruction="Do it", output="Done")
        original_dict = sample.model_dump()
        instruction_to_conversation(sample)
        assert sample.model_dump() == original_dict


# ══════════════════════════════════════════════
# conversation_to_instruction — unit tests
# ══════════════════════════════════════════════


class TestConversationToInstruction:

    def test_simple_pair(self):
        conv = ConversationSample(
            conversations=[
                ConversationTurn(role="user", content="Hi"),
                ConversationTurn(role="assistant", content="Hello!"),
            ]
        )
        raw = conversation_to_instruction(conv)
        assert isinstance(raw, RawSample)
        assert raw.instruction == "Hi"
        assert raw.output == "Hello!"
        assert raw.system_prompt is None

    def test_with_system(self):
        conv = ConversationSample(
            conversations=[
                ConversationTurn(role="system", content="Be helpful"),
                ConversationTurn(role="user", content="Q?"),
                ConversationTurn(role="assistant", content="A."),
            ]
        )
        raw = conversation_to_instruction(conv)
        assert raw.system_prompt == "Be helpful"
        assert raw.instruction == "Q?"
        assert raw.output == "A."

    def test_multi_turn_takes_last_assistant(self):
        conv = ConversationSample(
            conversations=[
                ConversationTurn(role="user", content="First Q"),
                ConversationTurn(role="assistant", content="First A"),
                ConversationTurn(role="user", content="Second Q"),
                ConversationTurn(role="assistant", content="Second A"),
            ]
        )
        raw = conversation_to_instruction(conv)
        assert raw.instruction == "First Q"
        assert raw.output == "Second A"

    def test_no_side_effects(self):
        conv = ConversationSample(
            conversations=[
                ConversationTurn(role="user", content="X"),
                ConversationTurn(role="assistant", content="Y"),
            ]
        )
        original_dict = conv.model_dump()
        conversation_to_instruction(conv)
        assert conv.model_dump() == original_dict


# ══════════════════════════════════════════════
# sharegpt_to_conversation — unit tests
# ══════════════════════════════════════════════


class TestSharegptToConversation:

    def test_basic(self):
        raw = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"},
            ]
        }
        conv = sharegpt_to_conversation(raw)
        assert conv.conversations[0].role == "user"
        assert conv.conversations[1].role == "assistant"

    def test_with_system(self):
        raw = {
            "conversations": [
                {"from": "system", "value": "Be kind"},
                {"from": "human", "value": "Hey"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        conv = sharegpt_to_conversation(raw)
        assert conv.conversations[0].role == "system"
        assert len(conv.conversations) == 3

    def test_missing_conversations_key(self):
        with pytest.raises(ValueError, match="missing"):
            sharegpt_to_conversation({})

    def test_unknown_role(self):
        raw = {
            "conversations": [
                {"from": "unknown_role", "value": "x"},
                {"from": "gpt", "value": "y"},
            ]
        }
        with pytest.raises(ValueError, match="Unknown ShareGPT role"):
            sharegpt_to_conversation(raw)

    def test_alternative_key_conversation(self):
        raw = {
            "conversation": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hey"},
            ]
        }
        conv = sharegpt_to_conversation(raw)
        assert len(conv.conversations) == 2


# ══════════════════════════════════════════════
# alpaca_to_conversation — unit tests
# ══════════════════════════════════════════════


class TestAlpacaToConversation:

    def test_basic(self):
        raw = {"instruction": "Sum 1+1", "output": "2"}
        conv = alpaca_to_conversation(raw)
        assert conv.conversations[0].role == "user"
        assert conv.conversations[0].content == "Sum 1+1"
        assert conv.conversations[1].role == "assistant"
        assert conv.conversations[1].content == "2"

    def test_with_input(self):
        raw = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
        conv = alpaca_to_conversation(raw)
        assert "Hello" in conv.conversations[0].content
        assert "Translate" in conv.conversations[0].content

    def test_missing_instruction(self):
        with pytest.raises(ValueError, match="instruction"):
            alpaca_to_conversation({"output": "x"})

    def test_missing_output(self):
        with pytest.raises(ValueError, match="output"):
            alpaca_to_conversation({"instruction": "x"})

    def test_empty_instruction(self):
        with pytest.raises(ValueError, match="instruction"):
            alpaca_to_conversation({"instruction": "", "output": "x"})


# ══════════════════════════════════════════════
# Hypothesis property-based: round-trip invariants
# ══════════════════════════════════════════════


class TestRoundTripProperties:
    """Property-based tests for converter round-trip invariants."""

    @given(sample=raw_samples())
    @settings(max_examples=100, deadline=None)
    def test_instruction_roundtrip_preserves_output(self, sample: RawSample):
        """instruction → conversation → instruction preserves output."""
        conv = instruction_to_conversation(sample)
        back = conversation_to_instruction(conv)
        assert back.output == sample.output

    @given(sample=raw_samples())
    @settings(max_examples=100, deadline=None)
    def test_instruction_roundtrip_preserves_system_prompt(
        self, sample: RawSample
    ):
        """instruction → conversation → instruction preserves system_prompt."""
        conv = instruction_to_conversation(sample)
        back = conversation_to_instruction(conv)
        assert back.system_prompt == sample.system_prompt

    @given(sample=raw_samples())
    @settings(max_examples=100, deadline=None)
    def test_instruction_roundtrip_no_input_preserves_instruction(
        self, sample: RawSample
    ):
        """When input is None, round-trip preserves instruction exactly."""
        assume(sample.input is None)
        conv = instruction_to_conversation(sample)
        back = conversation_to_instruction(conv)
        assert back.instruction == sample.instruction

    @given(sample=raw_samples())
    @settings(max_examples=100, deadline=None)
    def test_instruction_to_conversation_valid(self, sample: RawSample):
        """instruction → conversation always produces a valid ConversationSample."""
        conv = instruction_to_conversation(sample)
        assert isinstance(conv, ConversationSample)

    @given(conv=simple_conversations())
    @settings(max_examples=100, deadline=None)
    def test_conversation_roundtrip_preserves_instruction(
        self, conv: ConversationSample
    ):
        """conversation → instruction → conversation preserves user turn."""
        raw = conversation_to_instruction(conv)
        back = instruction_to_conversation(raw)
        assert back.conversations[0].content == conv.conversations[0].content

    @given(conv=simple_conversations())
    @settings(max_examples=100, deadline=None)
    def test_conversation_roundtrip_preserves_output(
        self, conv: ConversationSample
    ):
        """conversation → instruction → conversation preserves assistant turn."""
        raw = conversation_to_instruction(conv)
        back = instruction_to_conversation(raw)
        assert back.conversations[-1].content == conv.conversations[-1].content

    @given(conv=conversations_with_system())
    @settings(max_examples=100, deadline=None)
    def test_conversation_with_system_roundtrip(
        self, conv: ConversationSample
    ):
        """conversation(with system) → instruction → conversation preserves system."""
        raw = conversation_to_instruction(conv)
        back = instruction_to_conversation(raw)
        assert back.conversations[0].role == "system"
        assert back.conversations[0].content == conv.conversations[0].content

    @given(text=_text)
    @settings(max_examples=50, deadline=None)
    def test_alpaca_produces_valid_conversation(self, text: str):
        """Alpaca dict → ConversationSample always valid."""
        raw = {"instruction": text, "output": text}
        conv = alpaca_to_conversation(raw)
        assert isinstance(conv, ConversationSample)
        assert conv.conversations[-1].role == "assistant"

    @given(user_text=_text, assistant_text=_text)
    @settings(max_examples=50, deadline=None)
    def test_sharegpt_produces_valid_conversation(
        self, user_text: str, assistant_text: str
    ):
        """ShareGPT dict → ConversationSample always valid."""
        raw = {
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": assistant_text},
            ]
        }
        conv = sharegpt_to_conversation(raw)
        assert isinstance(conv, ConversationSample)


# ══════════════════════════════════════════════
# Purity checks
# ══════════════════════════════════════════════


class TestConverterPurity:
    """Verify converters are pure — no mutation of inputs."""

    def test_instruction_to_conversation_pure(self):
        sample = RawSample(instruction="A", output="B", system_prompt="C")
        dump_before = sample.model_dump()
        instruction_to_conversation(sample)
        assert sample.model_dump() == dump_before

    def test_conversation_to_instruction_pure(self):
        conv = ConversationSample(
            conversations=[
                ConversationTurn(role="user", content="X"),
                ConversationTurn(role="assistant", content="Y"),
            ]
        )
        dump_before = conv.model_dump()
        conversation_to_instruction(conv)
        assert conv.model_dump() == dump_before

    def test_sharegpt_pure(self):
        raw = {
            "conversations": [
                {"from": "human", "value": "a"},
                {"from": "gpt", "value": "b"},
            ]
        }
        import copy
        raw_copy = copy.deepcopy(raw)
        sharegpt_to_conversation(raw)
        assert raw == raw_copy

    def test_alpaca_pure(self):
        raw = {"instruction": "x", "output": "y"}
        import copy
        raw_copy = copy.deepcopy(raw)
        alpaca_to_conversation(raw)
        assert raw == raw_copy
