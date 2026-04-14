"""Load and validate strategies.yaml configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "capital": 100000,
    "port": 8088,
    "refresh_seconds": 300,
    "theme": "dark",
    "execution": {
        "cost_bps": 0,
        "max_abs_position": None,
    },
    "price_data": {
        "default_source": "abel",
        "default_timeframe": "1d",
    },
}

REQUIRED_STRATEGY_FIELDS = ("id", "name", "asset", "color", "engine", "trade_log")

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: str) -> str:
    """Replace ${ENV_VAR} patterns with environment variable values."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return _ENV_PATTERN.sub(_replace, value)


def _expand_env_recursive(obj: Any) -> Any:
    """Walk a nested dict/list and expand env vars in all string values."""
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_recursive(item) for item in obj]
    return obj


def _validate_strategy(strategy: dict, index: int) -> None:
    """Check that a strategy dict has all required fields."""
    for field in REQUIRED_STRATEGY_FIELDS:
        if field not in strategy:
            strat_label = strategy.get("id", strategy.get("name", f"index {index}"))
            raise ValueError(
                f"Strategy '{strat_label}' is missing required field '{field}'. "
                f"Required fields: {', '.join(REQUIRED_STRATEGY_FIELDS)}. "
                f"Add '{field}' to the strategy entry in strategies.yaml."
            )

    price_data = strategy.get("price_data")
    if price_data is not None:
        _validate_price_data(price_data, scope=f"strategy '{strategy.get('id', index)}'")

    paper_log = strategy.get("paper_log")
    if paper_log is not None and not isinstance(paper_log, str):
        raise ValueError(
            f"strategy '{strategy.get('id', index)}' paper_log must be a string path when provided."
        )


def _validate_price_data(price_data: Any, *, scope: str) -> None:
    if not isinstance(price_data, dict):
        raise ValueError(f"{scope} price_data must be a mapping.")

    source = price_data.get("source")
    if source is not None and source not in {"abel", "csv"}:
        raise ValueError(f"{scope} price_data.source must be 'abel' or 'csv', got '{source}'.")

    if source == "csv" and not price_data.get("path"):
        raise ValueError(f"{scope} price_data.path is required when source='csv'.")


def _validate_execution(execution: Any, *, scope: str) -> None:
    if not isinstance(execution, dict):
        raise ValueError(f"{scope} execution must be a mapping.")

    cost_bps = execution.get("cost_bps", 0)
    if cost_bps is not None and float(cost_bps) < 0:
        raise ValueError(f"{scope} execution.cost_bps must be >= 0.")

    max_abs_position = execution.get("max_abs_position")
    if max_abs_position is not None and float(max_abs_position) <= 0:
        raise ValueError(f"{scope} execution.max_abs_position must be > 0 when provided.")


def _merge_settings(user_settings: dict[str, Any]) -> dict[str, Any]:
    settings = {**DEFAULTS, **user_settings}
    settings["price_data"] = {
        **DEFAULTS.get("price_data", {}),
        **(user_settings.get("price_data") or {}),
    }
    settings["execution"] = {
        **DEFAULTS.get("execution", {}),
        **(user_settings.get("execution") or {}),
    }
    return settings


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load strategies.yaml configuration.

    Args:
        path: Path to YAML file. Defaults to strategies.yaml in current directory.

    Returns:
        Dict with keys 'settings' and 'strategies'.
    """
    if path is None:
        path = Path("strategies.yaml")
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}. "
            f"Run 'causal-edge init' to create a project, or specify --config."
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    if "strategies" not in raw:
        raise ValueError(
            "strategies.yaml must contain a 'strategies' key. Add 'strategies: []' at minimum."
        )

    user_settings = raw.get("settings") or {}
    settings = _merge_settings(user_settings)
    settings = _expand_env_recursive(settings)
    strategies = _expand_env_recursive(raw.get("strategies") or [])

    if "price_data" in settings:
        _validate_price_data(settings["price_data"], scope="settings")
    if "execution" in settings:
        _validate_execution(settings["execution"], scope="settings")

    for i, strat in enumerate(strategies):
        _validate_strategy(strat, i)

    return {"settings": settings, "strategies": strategies}
