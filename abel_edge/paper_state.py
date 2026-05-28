"""Small helpers for strategy-owned hosted paper state."""

from __future__ import annotations

from datetime import date, datetime
import json
import pickle
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pandas as pd

from abel_edge.runtime_paths import RuntimePaths, context_runtime_paths


class PaperStateStore:
    """Persist strategy-owned state under the runtime state directory.

    The store intentionally owns only path, serialization, and small paper
    signal metadata concerns. Strategy code still owns model fitting, feature
    construction, retrain calendars, and continuation semantics.
    """

    def __init__(
        self,
        root: Path | str,
        relative_path: Path | str = "strategy/paper_state.pkl",
        *,
        format: str | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.relative_path = _validate_relative_state_path(relative_path)
        self.path = self.root / self.relative_path
        self.format = _resolve_format(format, self.path)

    @classmethod
    def from_context(
        cls,
        context: Mapping[str, object] | None,
        relative_path: Path | str = "strategy/paper_state.pkl",
        *,
        format: str | None = None,
    ) -> "PaperStateStore":
        return cls(context_runtime_paths(context).state, relative_path, format=format)

    @classmethod
    def from_paths(
        cls,
        paths: RuntimePaths,
        relative_path: Path | str = "strategy/paper_state.pkl",
        *,
        format: str | None = None,
    ) -> "PaperStateStore":
        return cls(paths.state, relative_path, format=format)

    def as_of_key(self, value: Any) -> str | None:
        return paper_as_of_key(value)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self, default: Any = None) -> Any:
        if not self.path.is_file():
            return {} if default is None else default
        if self.format == "json":
            with self.path.open(encoding="utf-8") as handle:
                return json.load(handle)
        with self.path.open("rb") as handle:
            return pickle.load(handle)

    def save(self, payload: Any) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.format == "json":
            self.path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            with self.path.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return self.path

    def is_current(
        self,
        payload: Mapping[str, Any] | None,
        as_of: Any,
        *,
        key: str = "last_as_of",
    ) -> bool:
        if not isinstance(payload, Mapping):
            return False
        expected = self.as_of_key(as_of)
        actual = self.as_of_key(payload.get(key))
        return bool(expected and actual and expected == actual)

    def mark_current(
        self,
        payload: dict[str, Any],
        as_of: Any,
        *,
        key: str = "last_as_of",
    ) -> dict[str, Any]:
        marked = dict(payload)
        marked[key] = self.as_of_key(as_of)
        return marked

    def summary(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        as_of: Any = None,
    ) -> dict[str, Any]:
        state_as_of = self.as_of_key(as_of)
        if state_as_of is None and isinstance(payload, Mapping):
            state_as_of = self.as_of_key(
                payload.get("last_as_of")
                or payload.get("as_of")
                or payload.get("state_as_of")
            )
        summary = {
            "state_file": str(self.path),
            "state_path": self.relative_path.as_posix(),
        }
        if state_as_of:
            summary["state_as_of"] = state_as_of
        if isinstance(payload, Mapping):
            schema = payload.get("schema") or payload.get("state_schema")
            if isinstance(schema, str) and schema.strip():
                summary["state_schema"] = schema.strip()
        return summary

    def signal(
        self,
        *,
        next_position: Any,
        payload: Mapping[str, Any] | None = None,
        as_of: Any = None,
        **extras: Any,
    ) -> dict[str, Any]:
        signal = {
            "next_position": float(next_position),
            **self.summary(payload, as_of=as_of),
        }
        if signal.get("state_as_of"):
            signal["date"] = signal["state_as_of"]
        signal.update(
            {
                key: _safe_scalar(value)
                for key, value in extras.items()
                if _safe_scalar(value) is not None
            }
        )
        return signal


def paper_as_of_key(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
    try:
        timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _validate_relative_state_path(value: Path | str) -> PurePosixPath:
    text = str(value).replace("\\", "/").strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid paper state path: {text!r}")
    return path


def _resolve_format(format: str | None, path: Path) -> str:
    if format is not None:
        normalized = format.strip().lower()
        if normalized not in {"json", "pickle"}:
            raise ValueError("paper state format must be 'json' or 'pickle'")
        return normalized
    if path.suffix.lower() == ".json":
        return "json"
    return "pickle"


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


__all__ = ["PaperStateStore", "paper_as_of_key"]
