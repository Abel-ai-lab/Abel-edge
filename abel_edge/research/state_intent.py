"""State intent validation for hosted strategy promotion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


STATE_INTENT_SCHEMA = "abel-invest.state-intent/v1"
STATE_INTENT_ROLES = {
    "runtime_asset",
    "initial_state",
    "evidence",
    "exclude",
    "unknown",
}
REQUIRED_STATE_INTENT_FIELDS = {
    "path",
    "role",
    "mutableInPaper",
    "requiredForSignal",
}
DENYLISTED_STATE_INTENT_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "outputs",
    "rounds",
    "strategy_artifacts",
    "venv",
}
DENYLISTED_STATE_INTENT_SUFFIXES = {
    ".key",
    ".pem",
    ".pyc",
    ".pyo",
}


@dataclass(frozen=True)
class StateIntentEntry:
    path: str
    role: str
    mutable_in_paper: bool
    required_for_signal: bool
    produced_by: str


def validate_state_intent(
    payload: dict[str, Any] | None,
    *,
    root: Path | str,
) -> list[StateIntentEntry]:
    """Validate and normalize an abel-invest state intent payload."""

    if not payload:
        return []
    if not isinstance(payload, dict):
        raise ValueError("state intent must be an object")
    if payload.get("schema") != STATE_INTENT_SCHEMA:
        raise ValueError(
            f"state intent schema must be {STATE_INTENT_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("state intent entries must be a list")
    root_path = Path(root).resolve()
    normalized: list[StateIntentEntry] = []
    seen: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("state intent entries must be objects")
        missing = sorted(field for field in REQUIRED_STATE_INTENT_FIELDS if field not in raw_entry)
        if missing:
            raise ValueError(f"state intent entry missing fields: {', '.join(missing)}")
        path = validate_state_intent_path(raw_entry.get("path"))
        if path in seen:
            raise ValueError(f"duplicate state intent path: {path}")
        seen.add(path)
        role = str(raw_entry.get("role") or "").strip()
        if role not in STATE_INTENT_ROLES:
            raise ValueError(f"unsupported state intent role: {role!r}")
        _require_bool(raw_entry, "mutableInPaper")
        _require_bool(raw_entry, "requiredForSignal")
        source = root_path / Path(path)
        try:
            source.resolve().relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"state intent path escapes root: {path}") from exc
        if role != "exclude" and not source.is_file():
            raise ValueError(f"state intent file not found: {path}")
        normalized.append(
            StateIntentEntry(
                path=path,
                role=role,
                mutable_in_paper=bool(raw_entry["mutableInPaper"]),
                required_for_signal=bool(raw_entry["requiredForSignal"]),
                produced_by=str(raw_entry.get("producedBy") or "").strip(),
            )
        )
    return normalized


def validate_state_intent_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid state intent path: {text!r}")
    if any(part in DENYLISTED_STATE_INTENT_PARTS for part in path.parts):
        raise ValueError(f"denylisted state intent path: {text}")
    if path.suffix in DENYLISTED_STATE_INTENT_SUFFIXES:
        raise ValueError(f"denylisted state intent file: {text}")
    return path.as_posix()


def state_intent_from_manifest(manifest: dict[str, Any]) -> dict[str, Any] | None:
    payload = manifest.get("stateIntent")
    return payload if isinstance(payload, dict) else None


def _require_bool(payload: dict[str, Any], field: str) -> None:
    if not isinstance(payload.get(field), bool):
        raise ValueError(f"state intent field must be boolean: {field}")
