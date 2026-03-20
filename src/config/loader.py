"""Load and validate training configuration from YAML files."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from src.config.environment import EnvironmentResolver
from src.config.training import TrainingJobConfig

# Matches ${ssm:/some/path} with an optional :-default suffix
_SSM_PATTERN: re.Pattern[str] = re.compile(
    r"\$\{ssm:([^}:]+)(?::-([^}]*))?\}"
)


def load_training_config(path: str) -> TrainingJobConfig:
    """Load a YAML file and return a validated :class:`TrainingJobConfig`.

    Resolution order for string values:
      1. ``${ssm:/path/to/param}`` — AWS SSM Parameter Store look-ups
      2. ``${ENV_VAR}`` / ``${ENV_VAR:-default}`` — environment variables

    Args:
        path: Path to a YAML file whose structure matches
              :class:`TrainingJobConfig`.

    Returns:
        A fully validated Pydantic model.
    """
    raw = _load_yaml(path)
    resolved = _resolve_ssm(raw)
    resolved = EnvironmentResolver.resolve_dict(resolved)
    return TrainingJobConfig.model_validate(resolved)


def load_hpo_config(path: str) -> dict[str, Any]:
    """Load an HPO search-space YAML and resolve references.

    HPO configs are *not* validated against :class:`TrainingJobConfig`
    because they contain search-space definitions rather than concrete
    values.  Returns the raw (but reference-resolved) dictionary.
    """
    raw = _load_yaml(path)
    resolved = _resolve_ssm(raw)
    return EnvironmentResolver.resolve_dict(resolved)


# ── private helpers ─────────────────────────────────────────────


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(data).__name__}")
    return data


def _resolve_ssm(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve ``${ssm:/path}`` references in *data*."""
    resolved: dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, str):
            resolved[key] = _resolve_ssm_string(val)
        elif isinstance(val, dict):
            resolved[key] = _resolve_ssm(val)
        elif isinstance(val, list):
            resolved[key] = [
                _resolve_ssm_string(item) if isinstance(item, str) else item
                for item in val
            ]
        else:
            resolved[key] = val
    return resolved


def _resolve_ssm_string(value: str) -> str:
    """Replace ``${ssm:/path}`` tokens in *value*."""
    match = _SSM_PATTERN.search(value)
    if not match:
        return value

    def _replacer(m: re.Match[str]) -> str:
        ssm_path = m.group(1)
        default = m.group(2)
        try:
            return _get_ssm_parameter(ssm_path)
        except Exception:
            if default is not None:
                return default
            raise

    return _SSM_PATTERN.sub(_replacer, value)


def _get_ssm_parameter(name: str) -> str:
    """Fetch a single parameter from AWS SSM Parameter Store."""
    import boto3  # noqa: PLC0415

    client = boto3.client("ssm")
    response = client.get_parameter(Name=name, WithDecryption=True)
    return response["Parameter"]["Value"]
