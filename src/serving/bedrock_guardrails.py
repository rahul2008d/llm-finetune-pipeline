"""Bedrock Guardrails configuration and management."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

_DEFAULT_CONTENT_FILTERS = {
    "hate": "HIGH",
    "insults": "HIGH",
    "sexual": "HIGH",
    "violence": "HIGH",
    "misconduct": "HIGH",
}


class GuardrailsManager:
    """Create and manage Bedrock Guardrails for content safety."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with bedrock client.

        Args:
            region: AWS region for Bedrock operations.
        """
        if boto3 is None:
            raise ImportError("boto3 is required for GuardrailsManager")
        self.region = region
        self._client: Any = boto3.client("bedrock", region_name=region)

    def create_guardrail(
        self,
        name: str,
        description: str = "",
        content_filters: dict[str, str] | None = None,
        denied_topics: list[dict[str, str]] | None = None,
        word_filters: list[str] | None = None,
        pii_entity_types: list[str] | None = None,
        pii_action: str = "ANONYMIZE",
    ) -> str:
        """Create a Bedrock Guardrail.

        Args:
            name: Guardrail name.
            description: Guardrail description.
            content_filters: Dict mapping category to filter strength
                (NONE, LOW, MEDIUM, HIGH). Categories: hate, insults,
                sexual, violence, misconduct.
            denied_topics: List of topic dicts with name, definition,
                examples, type.
            word_filters: List of blocked words/phrases.
            pii_entity_types: PII types to filter (EMAIL, PHONE, SSN, etc.).
            pii_action: ANONYMIZE or BLOCK.

        Returns:
            The guardrail ID.
        """
        filters = content_filters or _DEFAULT_CONTENT_FILTERS

        create_kwargs: dict[str, Any] = {
            "name": name,
            "description": description or f"Guardrail: {name}",
            "blockedInputMessaging": "Your input was blocked by content safety filters.",
            "blockedOutputsMessaging": "The model output was blocked by content safety filters.",
        }

        # Content policy
        filter_configs: list[dict[str, str]] = []
        for category, strength in filters.items():
            filter_configs.append(
                {
                    "type": category.upper(),
                    "inputStrength": strength,
                    "outputStrength": strength,
                }
            )
        if filter_configs:
            create_kwargs["contentPolicyConfig"] = {
                "filtersConfig": filter_configs
            }

        # Topic policy
        if denied_topics:
            topic_configs: list[dict[str, Any]] = []
            for topic in denied_topics:
                topic_config: dict[str, Any] = {
                    "name": topic["name"],
                    "definition": topic.get("definition", ""),
                    "type": topic.get("type", "DENY"),
                }
                if "examples" in topic:
                    topic_config["examples"] = (
                        topic["examples"]
                        if isinstance(topic["examples"], list)
                        else [topic["examples"]]
                    )
                topic_configs.append(topic_config)
            create_kwargs["topicPolicyConfig"] = {
                "topicsConfig": topic_configs
            }

        # Word policy
        if word_filters:
            create_kwargs["wordPolicyConfig"] = {
                "wordsConfig": [{"text": w} for w in word_filters]
            }

        # PII / Sensitive information policy
        if pii_entity_types:
            create_kwargs["sensitiveInformationPolicyConfig"] = {
                "piiEntitiesConfig": [
                    {"type": entity, "action": pii_action}
                    for entity in pii_entity_types
                ]
            }

        response = self._client.create_guardrail(**create_kwargs)
        guardrail_id: str = response["guardrailId"]

        logger.info("Created guardrail", guardrail_id=guardrail_id, name=name)
        return guardrail_id

    def test_guardrail(
        self,
        guardrail_id: str,
        test_prompts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Test guardrail with sample prompts.

        Each test_prompt: {text, should_be_blocked: bool}.

        Args:
            guardrail_id: ID of the guardrail.
            test_prompts: List of test prompt dicts.

        Returns:
            Dict with passed, failed, total, false_positives,
            false_negatives, and details.
        """
        runtime_client: Any = boto3.client(
            "bedrock-runtime", region_name=self.region
        )

        passed = 0
        failed = 0
        false_positives = 0
        false_negatives = 0
        details: list[dict[str, Any]] = []

        for tp in test_prompts:
            text = tp["text"]
            should_block = tp.get("should_be_blocked", False)

            try:
                response = runtime_client.apply_guardrail(
                    guardrailIdentifier=guardrail_id,
                    guardrailVersion="DRAFT",
                    source="INPUT",
                    content=[{"text": {"text": text}}],
                )
                action = response.get("action", "")
                was_blocked = action == "GUARDRAIL_INTERVENED"

                if was_blocked == should_block:
                    passed += 1
                    test_status = "pass"
                else:
                    failed += 1
                    test_status = "fail"
                    if was_blocked and not should_block:
                        false_positives += 1
                    elif not was_blocked and should_block:
                        false_negatives += 1

                details.append(
                    {
                        "text": text,
                        "should_be_blocked": should_block,
                        "was_blocked": was_blocked,
                        "status": test_status,
                        "action": action,
                    }
                )
            except Exception as e:
                failed += 1
                details.append(
                    {
                        "text": text,
                        "should_be_blocked": should_block,
                        "was_blocked": False,
                        "status": f"error: {e}",
                        "action": "",
                    }
                )

        return {
            "passed": passed,
            "failed": failed,
            "total": len(test_prompts),
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "details": details,
        }

    def delete_guardrail(self, guardrail_id: str) -> None:
        """Delete a guardrail.

        Args:
            guardrail_id: ID of the guardrail to delete.
        """
        self._client.delete_guardrail(guardrailIdentifier=guardrail_id)
        logger.info("Deleted guardrail", guardrail_id=guardrail_id)

    def list_guardrails(self) -> list[dict[str, Any]]:
        """List all guardrails.

        Returns:
            List of guardrail summary dicts.
        """
        response = self._client.list_guardrails()
        guardrails: list[dict[str, Any]] = response.get("guardrails", [])
        logger.info("Listed guardrails", count=len(guardrails))
        return guardrails


__all__: list[str] = ["GuardrailsManager"]
