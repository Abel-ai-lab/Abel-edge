"""Credential and endpoint helpers for the optional Abel plugin."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_AUTH_BASE_URL = "https://api.abel.ai/echo"
DEFAULT_CAP_BASE_URL = "https://cap.abel.ai/api"


class MissingAbelApiKeyError(RuntimeError):
    """Raised when Abel-backed features are used without credentials."""


def _read_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in values:
            values[key] = value
    return values


def normalize_api_key(value: str | None) -> str:
    token = (value or "").strip()
    return token.removeprefix("Bearer ").strip()


def _format_env_value(value: str) -> str:
    if not value or any(char.isspace() for char in value) or "#" in value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def resolve_api_key(*, env_path: str | Path = ".env") -> str | None:
    env_values = _read_env_file(env_path)
    token = normalize_api_key(
        os.getenv("ABEL_API_KEY")
        or os.getenv("CAP_API_KEY")
        or env_values.get("ABEL_API_KEY")
        or env_values.get("CAP_API_KEY")
    )
    return token or None


def require_api_key(*, env_path: str | Path = ".env") -> str:
    token = resolve_api_key(env_path=env_path)
    if token:
        return token
    raise MissingAbelApiKeyError(
        "Abel API key not found. Set ABEL_API_KEY or CAP_API_KEY in your environment or .env "
        "before using Abel-backed features."
    )


def resolve_cap_base_url(*, env_path: str | Path = ".env") -> str:
    env_values = _read_env_file(env_path)
    configured = (os.getenv("ABEL_CAP_BASE_URL") or env_values.get("ABEL_CAP_BASE_URL") or "").strip()
    return (configured or DEFAULT_CAP_BASE_URL).rstrip("/")


def resolve_auth_base_url(*, env_path: str | Path = ".env") -> str:
    env_values = _read_env_file(env_path)
    configured = (
        os.getenv("ABEL_AUTH_BASE_URL") or env_values.get("ABEL_AUTH_BASE_URL") or ""
    ).strip()
    return (configured or DEFAULT_AUTH_BASE_URL).rstrip("/")


def persist_env_value(*, env_path: str | Path = ".env", key: str, value: str) -> None:
    path = Path(env_path)
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    assignment = f"{key}={_format_env_value(value)}"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            lines[index] = assignment
            break
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(assignment)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
