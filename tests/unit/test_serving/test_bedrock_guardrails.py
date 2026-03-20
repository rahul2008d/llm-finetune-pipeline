"""Tests for Bedrock Guardrails management."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.bedrock_guardrails import GuardrailsManager


@pytest.fixture()
def mock_bedrock_client() -> MagicMock:
    """Create a mock Bedrock client."""
    client = MagicMock()
    client.create_guardrail.return_value = {
        "guardrailId": "gr-test-123",
        "guardrailArn": "arn:aws:bedrock:us-east-1:123456789012:guardrail/gr-test-123",
    }
    client.list_guardrails.return_value = {
        "guardrails": [
            {"guardrailId": "gr-test-123", "name": "test-guardrail"}
        ]
    }
    return client


@pytest.fixture()
def guardrails_mgr(mock_bedrock_client: MagicMock) -> GuardrailsManager:
    """Create a GuardrailsManager with mocked boto3."""
    with patch("src.serving.bedrock_guardrails.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_bedrock_client
        mgr = GuardrailsManager(region="us-east-1")
    return mgr


class TestCreateGuardrail:
    """Tests for create_guardrail."""

    def test_create_guardrail_with_content_filters(
        self, guardrails_mgr: GuardrailsManager
    ) -> None:
        """Verify API call structure with content filters."""
        filters = {"hate": "HIGH", "violence": "MEDIUM"}

        guardrail_id = guardrails_mgr.create_guardrail(
            name="test-guardrail",
            content_filters=filters,
        )

        assert guardrail_id == "gr-test-123"
        call_kwargs = guardrails_mgr._client.create_guardrail.call_args[1]
        assert "contentPolicyConfig" in call_kwargs
        filter_configs = call_kwargs["contentPolicyConfig"]["filtersConfig"]
        filter_types = {f["type"] for f in filter_configs}
        assert "HATE" in filter_types
        assert "VIOLENCE" in filter_types

    def test_create_guardrail_with_pii(
        self, guardrails_mgr: GuardrailsManager
    ) -> None:
        """Verify PII config passed correctly."""
        guardrail_id = guardrails_mgr.create_guardrail(
            name="pii-guardrail",
            pii_entity_types=["EMAIL", "PHONE", "SSN"],
            pii_action="BLOCK",
        )

        assert guardrail_id == "gr-test-123"
        call_kwargs = guardrails_mgr._client.create_guardrail.call_args[1]
        assert "sensitiveInformationPolicyConfig" in call_kwargs
        pii_config = call_kwargs["sensitiveInformationPolicyConfig"][
            "piiEntitiesConfig"
        ]
        assert len(pii_config) == 3
        assert pii_config[0]["action"] == "BLOCK"

    def test_create_guardrail_with_word_filters(
        self, guardrails_mgr: GuardrailsManager
    ) -> None:
        """Verify word filters passed correctly."""
        guardrails_mgr.create_guardrail(
            name="word-guardrail",
            word_filters=["badword1", "badword2"],
        )

        call_kwargs = guardrails_mgr._client.create_guardrail.call_args[1]
        assert "wordPolicyConfig" in call_kwargs
        words = call_kwargs["wordPolicyConfig"]["wordsConfig"]
        assert len(words) == 2

    def test_create_guardrail_with_denied_topics(
        self, guardrails_mgr: GuardrailsManager
    ) -> None:
        """Verify denied topics config."""
        topics = [
            {
                "name": "financial-advice",
                "definition": "Providing specific investment advice",
                "type": "DENY",
            }
        ]
        guardrails_mgr.create_guardrail(
            name="topic-guardrail",
            denied_topics=topics,
        )

        call_kwargs = guardrails_mgr._client.create_guardrail.call_args[1]
        assert "topicPolicyConfig" in call_kwargs


class TestTestGuardrail:
    """Tests for test_guardrail."""

    def test_test_guardrail_reports_results(
        self, guardrails_mgr: GuardrailsManager
    ) -> None:
        """Verify pass/fail counting."""
        mock_runtime = MagicMock()
        # First call: blocked, second call: allowed
        mock_runtime.apply_guardrail.side_effect = [
            {"action": "GUARDRAIL_INTERVENED"},
            {"action": "NONE"},
        ]

        with patch("src.serving.bedrock_guardrails.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_runtime

            result = guardrails_mgr.test_guardrail(
                guardrail_id="gr-test-123",
                test_prompts=[
                    {"text": "Harmful content", "should_be_blocked": True},
                    {"text": "Normal content", "should_be_blocked": False},
                ],
            )

        assert result["passed"] == 2
        assert result["failed"] == 0
        assert result["total"] == 2
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0


class TestDeleteGuardrail:
    """Tests for delete_guardrail."""

    def test_delete_guardrail(
        self, guardrails_mgr: GuardrailsManager
    ) -> None:
        """Verify API call."""
        guardrails_mgr.delete_guardrail("gr-test-123")

        guardrails_mgr._client.delete_guardrail.assert_called_once_with(
            guardrailIdentifier="gr-test-123"
        )


class TestListGuardrails:
    """Tests for list_guardrails."""

    def test_list_guardrails(
        self, guardrails_mgr: GuardrailsManager
    ) -> None:
        """Verify list returns guardrails."""
        result = guardrails_mgr.list_guardrails()
        assert len(result) == 1
        assert result[0]["name"] == "test-guardrail"
