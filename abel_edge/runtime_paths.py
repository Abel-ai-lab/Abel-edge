"""Runtime path contract helpers for hosted strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


ABEL_BASE_STRATEGY_DIR = "ABEL_BASE_STRATEGY_DIR"
ABEL_RUNTIME_DIR = "ABEL_RUNTIME_DIR"
ABEL_STATE_DIR = "ABEL_STATE_DIR"


@dataclass(frozen=True)
class RuntimePaths:
    """Normalized runtime paths exposed to strategy code."""

    base_strategy: Path
    runtime: Path
    state: Path

    def as_context_payload(self) -> dict[str, str]:
        return {
            "base_strategy": str(self.base_strategy),
            "runtime": str(self.runtime),
            "state": str(self.state),
        }


def runtime_paths(
    *,
    env: Mapping[str, str] | None = None,
    base_strategy: Path | str | None = None,
    runtime: Path | str | None = None,
    state: Path | str | None = None,
    create: bool = False,
) -> RuntimePaths:
    """Return runtime paths from explicit overrides or runner environment.

    Explicit arguments are useful for local evaluate/replay. Hosted runner paths
    should come from environment variables.
    """

    source = os.environ if env is None else env
    paths = RuntimePaths(
        base_strategy=_resolve_runtime_path(
            base_strategy if base_strategy is not None else source.get(ABEL_BASE_STRATEGY_DIR),
            ABEL_BASE_STRATEGY_DIR,
        ),
        runtime=_resolve_runtime_path(
            runtime if runtime is not None else source.get(ABEL_RUNTIME_DIR),
            ABEL_RUNTIME_DIR,
        ),
        state=_resolve_runtime_path(
            state if state is not None else source.get(ABEL_STATE_DIR),
            ABEL_STATE_DIR,
        ),
    )
    if create:
        paths.base_strategy.mkdir(parents=True, exist_ok=True)
        paths.runtime.mkdir(parents=True, exist_ok=True)
        paths.state.mkdir(parents=True, exist_ok=True)
    return paths


def context_runtime_paths(context: Mapping[str, object] | None) -> RuntimePaths:
    """Build RuntimePaths from StrategyEngine.context payload or env vars."""

    payload = (context or {}).get("_runtime_paths") if context is not None else None
    if isinstance(payload, Mapping):
        return runtime_paths(
            base_strategy=payload.get("base_strategy"),
            runtime=payload.get("runtime"),
            state=payload.get("state"),
        )
    return runtime_paths()


def inject_runtime_paths(
    context: dict,
    paths: RuntimePaths,
) -> dict:
    """Return a context copy with normalized runtime paths attached."""

    updated = dict(context)
    updated["_runtime_paths"] = paths.as_context_payload()
    return updated


def _resolve_runtime_path(value: Path | str | object, env_name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"Missing runtime path: {env_name}")
    return Path(text).expanduser().resolve()
