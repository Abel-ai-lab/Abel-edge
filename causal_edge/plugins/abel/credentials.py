"""Credential and endpoint helpers for the optional Abel plugin."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_AUTH_BASE_URL = "https://api.abel.ai/echo"
DEFAULT_CAP_BASE_URL = "https://cap.abel.ai/api"
CAUSAL_ABEL_ENV_BASENAMES = (".env.skill", ".env.skills", ".env")


def _global_causal_abel_skill_dirs() -> tuple[Path, ...]:
    home_dir = Path.home()
    return (
        home_dir / ".config" / "opencode" / "skills" / "causal-abel",
        home_dir / ".codex" / "skills" / "causal-abel",
    )


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


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _resolve_env_path(path: str | Path) -> Path:
    env_path = Path(path).expanduser()
    if env_path.is_absolute():
        return env_path
    return (Path.cwd() / env_path).resolve()


def _find_causal_abel_skill_dir(*, env_path: str | Path = ".env") -> Path | None:
    start_dir = _resolve_env_path(env_path).parent
    for directory in (start_dir, *start_dir.parents):
        skill_dir = directory / ".agents" / "skills" / "causal-abel"
        if skill_dir.is_dir():
            return skill_dir
    return None


def _candidate_shared_auth_files(*, env_path: str | Path = ".env") -> list[Path]:
    candidates: list[Path] = []
    explicit_auth_file = (os.getenv("ABEL_AUTH_ENV_FILE") or "").strip()
    if explicit_auth_file:
        candidates.append(_resolve_env_path(explicit_auth_file))

    skill_dir = _find_causal_abel_skill_dir(env_path=env_path)
    if skill_dir is not None:
        for basename in CAUSAL_ABEL_ENV_BASENAMES:
            candidates.append(skill_dir / basename)

    for skill_dir in _global_causal_abel_skill_dirs():
        if not skill_dir.is_dir():
            continue
        for basename in CAUSAL_ABEL_ENV_BASENAMES:
            candidates.append(skill_dir / basename)

    return _dedupe_paths(candidates)


def _resolve_token_from_values(values: dict[str, str]) -> str | None:
    token = normalize_api_key(values.get("ABEL_API_KEY") or values.get("CAP_API_KEY"))
    return token or None


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
    if token:
        return token

    for shared_auth_file in _candidate_shared_auth_files(env_path=env_path):
        shared_token = _resolve_token_from_values(_read_env_file(shared_auth_file))
        if shared_token:
            return shared_token

    return None


def require_api_key(*, env_path: str | Path = ".env") -> str:
    token = resolve_api_key(env_path=env_path)
    if token:
        return token
    shared_auth_hint = (
        "Set ABEL_AUTH_ENV_FILE to a causal-abel auth file if you store it outside the project."
    )
    skill_dir = _find_causal_abel_skill_dir(env_path=env_path)
    if skill_dir is not None:
        shared_auth_hint = (
            "`causal-abel` appears to be installed. Run "
            f"`python {skill_dir / 'scripts' / 'cap_probe.py'} auth-status --compact` to confirm auth, "
            "or set ABEL_AUTH_ENV_FILE to the exported auth file."
        )
    raise MissingAbelApiKeyError(
        "Abel API key not found. Install the `causal-abel` skill and complete its OAuth flow, "
        "or set ABEL_API_KEY or CAP_API_KEY in your environment or project .env before using Abel-backed "
        "features. causal-edge also checks ABEL_AUTH_ENV_FILE, project-local "
        ".agents/skills/causal-abel/.env.skill, and known global skill installs such as "
        "~/.config/opencode/skills/causal-abel/.env.skill or ~/.codex/skills/causal-abel/.env.skill. "
        f"{shared_auth_hint}"
    )


def resolve_cap_base_url(*, env_path: str | Path = ".env") -> str:
    env_values = _read_env_file(env_path)
    configured = (
        os.getenv("ABEL_CAP_BASE_URL") or env_values.get("ABEL_CAP_BASE_URL") or ""
    ).strip()
    return (configured or DEFAULT_CAP_BASE_URL).rstrip("/")


def resolve_auth_base_url(*, env_path: str | Path = ".env") -> str:
    env_values = _read_env_file(env_path)
    configured = (
        os.getenv("ABEL_AUTH_BASE_URL") or env_values.get("ABEL_AUTH_BASE_URL") or ""
    ).strip()
    return (configured or DEFAULT_AUTH_BASE_URL).rstrip("/")


def persist_env_value(*, env_path: str | Path = ".env", key: str, value: str) -> None:
    path = Path(env_path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
