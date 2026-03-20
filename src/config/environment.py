"""Environment variable resolution with support for defaults and overrides."""

import os
import re
from typing import Any


class EnvironmentResolver:
    """Resolve environment variable references in configuration values.

    Supports ``${VAR_NAME}`` and ``${VAR_NAME:-default}`` syntax.
    """

    _PATTERN: re.Pattern[str] = re.compile(r"\$\{([^}]+)\}")

    @staticmethod
    def resolve(value: str) -> str:
        """Resolve environment variable placeholders in a string.

        Args:
            value: String potentially containing ``${VAR}`` or ``${VAR:-default}`` patterns.

        Returns:
            String with all environment variable references resolved.

        Raises:
            KeyError: If an environment variable is not set and no default is provided.
        """

        def _replace(match: re.Match[str]) -> str:
            expr = match.group(1)
            if ":-" in expr:
                var_name, default = expr.split(":-", 1)
                return os.environ.get(var_name.strip(), default.strip())
            var_name = expr.strip()
            env_value = os.environ.get(var_name)
            if env_value is None:
                raise KeyError(f"Environment variable '{var_name}' is not set")
            return env_value

        return EnvironmentResolver._PATTERN.sub(_replace, value)

    @staticmethod
    def resolve_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Recursively resolve environment variables in a dictionary.

        Args:
            data: Dictionary with string values that may contain env var references.

        Returns:
            New dictionary with all string values resolved.
        """
        resolved: dict[str, Any] = {}
        for key, val in data.items():
            if isinstance(val, str):
                resolved[key] = EnvironmentResolver.resolve(val)
            elif isinstance(val, dict):
                resolved[key] = EnvironmentResolver.resolve_dict(val)
            elif isinstance(val, list):
                resolved[key] = [
                    EnvironmentResolver.resolve(item) if isinstance(item, str) else item
                    for item in val
                ]
            else:
                resolved[key] = val
        return resolved


__all__: list[str] = ["EnvironmentResolver"]
