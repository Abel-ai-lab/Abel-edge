"""Load and validate strategies.yaml configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from causal_edge.engine.adapter_registry import (
    AdapterRegistryError,
    ensure_adapter_registered,
    load_adapter_imports,
)

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
        "default_adapter": "abel",
        "default_timeframe": "1d",
    },
    "data_contract": {
        "profile": "daily",
    },
    "data_adapters": {
        "imports": [],
    },
}

REQUIRED_STRATEGY_FIELDS = ("id", "name", "asset", "color", "engine", "trade_log")

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
DEFAULT_CONFIG_PATH = Path("strategies.yaml")
LOCAL_CONFIG_PATH = Path("strategies.local.yaml")


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

    feeds = strategy.get("feeds")
    if feeds is not None:
        _validate_feeds(feeds, scope=f"strategy '{strategy.get('id', index)}'")


def _validate_price_data(price_data: Any, *, scope: str) -> None:
    if not isinstance(price_data, dict):
        raise ValueError(f"{scope} price_data must be a mapping.")

    adapter = price_data.get("adapter") or price_data.get("source")
    if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
        raise ValueError(f"{scope} price_data.adapter must be a non-empty string when provided.")

    if str(adapter or "").strip().lower() == "csv" and not price_data.get("path"):
        raise ValueError(f"{scope} price_data.path is required when adapter='csv'.")


def _validate_execution(execution: Any, *, scope: str) -> None:
    if not isinstance(execution, dict):
        raise ValueError(f"{scope} execution must be a mapping.")

    cost_bps = execution.get("cost_bps", 0)
    if cost_bps is not None and float(cost_bps) < 0:
        raise ValueError(f"{scope} execution.cost_bps must be >= 0.")

    max_abs_position = execution.get("max_abs_position")
    if max_abs_position is not None and float(max_abs_position) <= 0:
        raise ValueError(f"{scope} execution.max_abs_position must be > 0 when provided.")


def _validate_data_contract(data_contract: Any, *, scope: str) -> None:
    if not isinstance(data_contract, dict):
        raise ValueError(f"{scope} data_contract must be a mapping.")
    profile = str(data_contract.get("profile", "daily")).strip().lower()
    if profile != "daily":
        raise ValueError(f"{scope} data_contract.profile must be 'daily', got '{profile}'.")


def _validate_data_adapters(data_adapters: Any, *, scope: str) -> None:
    if not isinstance(data_adapters, dict):
        raise ValueError(f"{scope} data_adapters must be a mapping.")
    imports = data_adapters.get("imports", [])
    if not isinstance(imports, list):
        raise ValueError(f"{scope} data_adapters.imports must be a list.")
    for module_name in imports:
        if not isinstance(module_name, str) or not module_name.strip():
            raise ValueError(f"{scope} data_adapters.imports must contain non-empty strings.")


def _validate_feeds(feeds: Any, *, scope: str) -> None:
    if not isinstance(feeds, dict):
        raise ValueError(f"{scope} feeds must be a mapping.")
    if "primary" in feeds:
        raise ValueError(f"{scope} feeds.primary is reserved and may not be declared explicitly.")
    for name, feed in feeds.items():
        if not isinstance(feed, dict):
            raise ValueError(f"{scope} feed '{name}' must be a mapping.")
        kind = str(feed.get("kind", "")).strip().lower()
        if kind not in {"bars", "series"}:
            raise ValueError(f"{scope} feed '{name}' kind must be 'bars' or 'series'.")
        adapter = feed.get("adapter") or feed.get("source")
        if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
            raise ValueError(f"{scope} feed '{name}' adapter must be a non-empty string.")
        if str(adapter or "").strip().lower() == "csv" and not feed.get("path"):
            raise ValueError(f"{scope} feed '{name}' path is required when adapter='csv'.")
        if kind == "series" and not feed.get("field"):
            raise ValueError(f"{scope} feed '{name}' field is required when kind='series'.")


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
    settings["data_contract"] = {
        **DEFAULTS.get("data_contract", {}),
        **(user_settings.get("data_contract") or {}),
    }
    settings["data_adapters"] = {
        **DEFAULTS.get("data_adapters", {}),
        **(user_settings.get("data_adapters") or {}),
    }
    return settings


def _normalize_feed(
    name: str,
    feed: dict[str, Any],
    *,
    settings: dict[str, Any],
    strategy: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    price_settings = settings.get("price_data") or {}
    normalized = dict(feed)
    normalized["name"] = name
    normalized["kind"] = str(feed.get("kind", "")).strip().lower()
    normalized["adapter"] = feed.get("adapter") or feed.get("source") or price_settings.get(
        "default_adapter",
        price_settings.get("default_source", "abel"),
    )
    normalized["timeframe"] = (
        feed.get("timeframe") or price_settings.get("default_timeframe", "1d")
    )
    normalized["profile"] = profile
    if normalized["kind"] == "bars":
        if feed.get("symbol"):
            normalized["symbol"] = feed["symbol"]
        elif not feed.get("symbols"):
            normalized.setdefault("symbol", strategy.get("asset"))
    if normalized["kind"] == "series" and feed.get("symbol"):
        normalized["symbol"] = feed["symbol"]
    if normalized["adapter"] == "csv" and feed.get("path"):
        normalized["path"] = feed["path"]
    if normalized["kind"] == "series":
        normalized["field"] = feed["field"]
    return normalized


def _synthesized_primary_feed(
    strategy: dict[str, Any],
    *,
    settings: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    price_settings = settings.get("price_data") or {}
    strategy_price = strategy.get("price_data") or {}
    adapter = strategy_price.get("adapter") or strategy_price.get("source") or price_settings.get(
        "default_adapter",
        price_settings.get("default_source", "abel"),
    )
    timeframe = strategy_price.get("timeframe") or price_settings.get("default_timeframe", "1d")
    primary = {
        "name": "primary",
        "kind": "bars",
        "adapter": adapter,
        "timeframe": timeframe,
        "symbol": strategy_price.get("symbol") or strategy.get("asset"),
        "profile": profile,
    }
    for key, value in strategy_price.items():
        if key in {"adapter", "source", "timeframe", "symbol", "path", "env_path"}:
            continue
        if value is not None:
            primary[key] = value
    if adapter == "csv" and strategy_price.get("path"):
        primary["path"] = strategy_price["path"]
    if strategy_price.get("env_path"):
        primary["env_path"] = strategy_price["env_path"]
    return primary


def _normalize_strategy_runtime(strategy: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(strategy)
    profile = str((settings.get("data_contract") or {}).get("profile", "daily")).strip().lower()
    feeds = {"primary": _synthesized_primary_feed(strategy, settings=settings, profile=profile)}
    for name, feed in ((strategy.get("feeds") or {}).items()):
        feeds[name] = _normalize_feed(
            name,
            feed,
            settings=settings,
            strategy=strategy,
            profile=profile,
        )
    normalized["_feeds"] = feeds
    normalized["_data_contract"] = {"profile": profile}
    return normalized


def _validate_declared_adapters(strategies: list[dict[str, Any]]) -> None:
    for strategy in strategies:
        for feed_name, feed_cfg in (strategy.get("_feeds") or {}).items():
            try:
                ensure_adapter_registered(feed_cfg["adapter"])
            except AdapterRegistryError as exc:
                raise ValueError(
                    f"strategy '{strategy['id']}' feed '{feed_name}': {exc}"
                ) from exc


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Resolve the strategies config path with local override support."""
    if path is None:
        if LOCAL_CONFIG_PATH.exists():
            return LOCAL_CONFIG_PATH
        return DEFAULT_CONFIG_PATH
    return Path(path)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load strategies configuration.

    Args:
        path: Explicit path to YAML file. When omitted, prefers
            strategies.local.yaml in the current directory and falls back to
            strategies.yaml.

    Returns:
        Dict with keys 'settings' and 'strategies'.
    """
    path = resolve_config_path(path)

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
    if "data_contract" in settings:
        _validate_data_contract(settings["data_contract"], scope="settings")
    if "data_adapters" in settings:
        _validate_data_adapters(settings["data_adapters"], scope="settings")

    for i, strat in enumerate(strategies):
        _validate_strategy(strat, i)

    normalized_strategies = [_normalize_strategy_runtime(strat, settings) for strat in strategies]
    load_adapter_imports((settings.get("data_adapters") or {}).get("imports"))
    _validate_declared_adapters(normalized_strategies)
    return {"settings": settings, "strategies": normalized_strategies}
