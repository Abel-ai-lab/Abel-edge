"""Adapter registry for framework-managed feed loading."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from causal_edge.engine.cache import (
    cache_covers_request,
    cache_entry_for_request,
    load_cached_bars,
    load_cached_metadata,
    write_cached_bars,
)
from causal_edge.engine.feed_contract import FeedContractError


@dataclass(frozen=True)
class FeedLoadRequest:
    adapter: str
    kind: str
    symbol: str | None
    field: str | None
    timeframe: str | None
    start: object | None
    end: object | None
    limit: int | None
    profile: str
    options: dict[str, object]
    strategy_id: str | None
    feed_name: str


class DataFeedAdapter(Protocol):
    assume_utc_for_naive: bool

    def load(self, request: FeedLoadRequest) -> pd.DataFrame: ...


_ADAPTERS: dict[str, DataFeedAdapter] = {}
_BUILTINS_REGISTERED = False


class AdapterRegistryError(FeedContractError):
    """Raised when a declared adapter cannot be resolved."""


def register_adapter(name: str, adapter: DataFeedAdapter) -> None:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise AdapterRegistryError("Adapter name must be a non-empty string.")
    _ADAPTERS[normalized] = adapter


def load_adapter_imports(imports: list[str] | None) -> None:
    ensure_builtin_adapters()
    for module_name in imports or []:
        if not isinstance(module_name, str) or not module_name.strip():
            raise AdapterRegistryError("settings.data_adapters.imports must contain module strings.")
        importlib.import_module(module_name.strip())


def ensure_adapter_registered(name: str) -> None:
    resolve_adapter(name)


def resolve_adapter(name: str) -> DataFeedAdapter:
    ensure_builtin_adapters()
    normalized = str(name or "").strip().lower()
    adapter = _ADAPTERS.get(normalized)
    if adapter is None:
        raise AdapterRegistryError(
            f"Adapter '{name}' is not registered. "
            "Declare it via settings.data_adapters.imports or use a built-in adapter."
        )
    return adapter


def ensure_builtin_adapters() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    register_adapter("csv", CSVDataFeedAdapter())
    register_adapter("abel", AbelDataFeedAdapter())
    _BUILTINS_REGISTERED = True


class CSVDataFeedAdapter:
    assume_utc_for_naive = True

    def load(self, request: FeedLoadRequest) -> pd.DataFrame:
        path_value = request.options.get("path")
        if not path_value:
            raise AdapterRegistryError(
                f"Feed '{request.feed_name}' uses adapter='csv' but is missing 'path'."
            )
        path = Path(path_value)
        df = pd.read_csv(path)
        if request.kind == "bars":
            return _csv_bars_frame(df, request)
        if request.kind == "series":
            return _csv_series_frame(df, request)
        raise AdapterRegistryError(
            f"Feed '{request.feed_name}' declares unsupported kind '{request.kind}'."
        )


class AbelDataFeedAdapter:
    assume_utc_for_naive = False

    def load(self, request: FeedLoadRequest) -> pd.DataFrame:
        symbol = request.symbol
        if not symbol:
            raise AdapterRegistryError(
                f"Feed '{request.feed_name}' uses adapter='abel' but is missing 'symbol'."
            )
        try:
            credentials_module = importlib.import_module("causal_edge.plugins.abel.credentials")
            prices_module = importlib.import_module("causal_edge.plugins.abel.prices")
            missing_api_key_error = credentials_module.MissingAbelApiKeyError
            fetch_bars = prices_module.fetch_bars
        except ImportError as exc:
            raise AdapterRegistryError(
                "Abel adapter is unavailable. See: causal_edge/plugins/AGENTS.md"
            ) from exc

        fields: list[str] | None = None
        if request.kind == "series":
            fields = [request.field or "close"]
        elif request.kind == "bars":
            raw_fields = request.options.get("fields")
            if isinstance(raw_fields, list):
                fields = [str(field) for field in raw_fields]

        entry = cache_entry_for_request(
            adapter=request.adapter,
            symbol=symbol,
            timeframe=request.timeframe,
            profile=request.profile,
            options=request.options,
            cache_root=request.options.get("cache_root"),
        )
        cached_metadata = load_cached_metadata(entry)
        if cache_covers_request(
            cached_metadata,
            start=request.start,
            end=request.end,
        ):
            cached = load_cached_bars(entry)
            if cached is not None:
                bars = cached
            else:
                bars = pd.DataFrame()
        else:
            bars = pd.DataFrame()

        if bars.empty:
            effective_limit = request.limit
            if effective_limit is None or effective_limit < 5000:
                effective_limit = 5000
            try:
                bars = fetch_bars(
                    symbols=[symbol],
                    start=request.start,
                    end=request.end,
                    timeframe=request.timeframe or "1d",
                    limit=effective_limit,
                    fields=fields,
                    config=request.options,
                )
                write_cached_bars(entry, bars)
            except missing_api_key_error as exc:
                raise AdapterRegistryError(str(exc)) from exc

        if request.kind == "bars":
            return bars

        field = request.field or "close"
        if field not in bars.columns:
            raise AdapterRegistryError(
                f"Feed '{request.feed_name}' could not resolve field '{field}' from adapter 'abel'."
            )
        frame = bars[["timestamp", field]].copy()
        if "symbol" in bars.columns:
            frame["symbol"] = bars["symbol"]
        return frame.rename(columns={field: "value"})


def _csv_bars_frame(df: pd.DataFrame, request: FeedLoadRequest) -> pd.DataFrame:
    frame = df.copy()
    if "symbol" not in frame.columns:
        if not request.symbol:
            raise AdapterRegistryError(
                f"CSV bars feed '{request.feed_name}' must include 'symbol' or declare one in config."
            )
        frame["symbol"] = request.symbol
    return frame


def _csv_series_frame(df: pd.DataFrame, request: FeedLoadRequest) -> pd.DataFrame:
    frame = df.copy()
    field = request.field or "value"
    if field in frame.columns:
        frame = frame.rename(columns={field: "value"})
    elif "value" not in frame.columns:
        raise AdapterRegistryError(
            f"CSV series feed '{request.feed_name}' is missing declared field '{field}'."
        )
    if "symbol" not in frame.columns and request.symbol:
        frame["symbol"] = request.symbol
    return frame
