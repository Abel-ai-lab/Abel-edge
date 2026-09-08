"""Adapter registry for framework-managed feed loading."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from abel_edge.engine.cache import (
    cache_covers_request,
    cache_entry_for_request,
    load_cached_bars,
    load_cached_metadata,
    load_cached_point_in_time_series,
    point_in_time_cache_covers_request,
    point_in_time_cache_entry,
    write_cached_bars,
    write_cached_point_in_time_series,
)
from abel_edge.engine.feed_contract import (
    FeedContractError,
    apply_max_data_date_guard,
    assert_frame_respects_max_data_date,
)
from abel_edge.engine.point_in_time_series import (
    PointInTimeSeriesSpec,
    is_date_only_bound,
)

ABEL_BAR_FIELDS = ["open", "high", "low", "close", "volume"]
ABEL_BAR_CACHE_COLUMNS = ["timestamp", "symbol", *ABEL_BAR_FIELDS]


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
    series_spec: PointInTimeSeriesSpec | None = None


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
        apply_max_data_date_guard(
            request.end,
            source=f"feed '{request.feed_name}' adapter request",
        )
        path_value = request.options.get("path")
        if request.series_spec is not None:
            path_value = request.series_spec.source_request.get("path") or path_value
        if request.kind == "point_in_time_series":
            if request.series_spec is None:
                raise AdapterRegistryError(
                    f"Feed '{request.feed_name}' is missing its point-in-time series spec."
                )
            if request.series_spec.payload.get("transforms"):
                raise AdapterRegistryError(
                    "The CSV point-in-time adapter does not support "
                    "series_spec.transforms; use an adapter that materializes the "
                    "declared transforms."
                )
        if not path_value:
            raise AdapterRegistryError(
                f"Feed '{request.feed_name}' uses adapter='csv' but is missing 'path'."
            )
        path = Path(path_value)
        df = pd.read_csv(path)
        if request.kind == "bars":
            frame = _csv_bars_frame(df, request)
            assert_frame_respects_max_data_date(frame, source=f"feed '{request.feed_name}'")
            return frame
        if request.kind == "series":
            frame = _csv_series_frame(df, request)
            assert_frame_respects_max_data_date(frame, source=f"feed '{request.feed_name}'")
            return frame
        if request.kind == "point_in_time_series":
            df.attrs["source_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            df.attrs["series_spec_sha256"] = request.series_spec.sha256
            return df
        raise AdapterRegistryError(
            f"Feed '{request.feed_name}' declares unsupported kind '{request.kind}'."
        )


class AbelDataFeedAdapter:
    assume_utc_for_naive = False

    def load(self, request: FeedLoadRequest) -> pd.DataFrame:
        guarded_end = apply_max_data_date_guard(
            request.end,
            source=f"feed '{request.feed_name}' adapter request",
        )
        if request.kind == "point_in_time_series":
            if request.series_spec is None:
                raise AdapterRegistryError(
                    f"Feed '{request.feed_name}' is missing its point-in-time series spec."
                )
            try:
                canonical_module = importlib.import_module(
                    "abel_edge.plugins.abel.cap_node_series"
                )
                credentials_module = importlib.import_module(
                    "abel_edge.plugins.abel.credentials"
                )
            except ImportError as exc:
                raise AdapterRegistryError(
                    "Abel canonical-node data support is unavailable. "
                    "See: abel_edge/plugins/AGENTS.md"
                ) from exc
            # A live request is reusable only after its effective upper bound
            # has elapsed; today's inclusive date is incomplete until tomorrow.
            cache_root = (
                request.options.get("cache_root")
                if _point_in_time_cache_end_has_elapsed(guarded_end)
                else None
            )
            entry = None
            if cache_root:
                source_identity = credentials_module.resolve_cap_base_url(
                    env_path=request.options.get("env_path", ".env")
                )
                entry = point_in_time_cache_entry(
                    adapter=request.adapter,
                    series_spec_sha256=point_in_time_cache_identity(
                        series_spec_sha256=request.series_spec.sha256,
                        source_identity=source_identity,
                    ),
                    cache_root=cache_root,
                )
                metadata = load_cached_metadata(entry)
                if point_in_time_cache_covers_request(
                    metadata,
                    series_spec_sha256=request.series_spec.sha256,
                    start=request.start,
                    end=guarded_end,
                    limit=request.limit,
                ):
                    cached = load_cached_point_in_time_series(
                        entry,
                        metadata=metadata,
                    )
                    if cached is not None:
                        return cached
            frame = canonical_module.load_cap_node_series(
                series_spec=request.series_spec,
                start=request.start,
                end=guarded_end,
                limit=request.limit,
                config={
                    "env_path": request.options.get("env_path") or ".env",
                },
            )
            if entry is not None:
                write_cached_point_in_time_series(
                    entry,
                    frame,
                    series_spec_sha256=request.series_spec.sha256,
                    source_receipt_sha256=str(
                        frame.attrs.get("source_receipt_sha256") or ""
                    ),
                    requested_start=request.start,
                    requested_end=guarded_end,
                    requested_limit=request.limit,
                )
            return frame

        symbol = request.symbol
        if not symbol:
            raise AdapterRegistryError(
                f"Feed '{request.feed_name}' uses adapter='abel' but is missing 'symbol'."
            )
        try:
            credentials_module = importlib.import_module("abel_edge.plugins.abel.credentials")
            prices_module = importlib.import_module("abel_edge.plugins.abel.prices")
            missing_api_key_error = credentials_module.MissingAbelApiKeyError
            fetch_bars = prices_module.fetch_bars
        except ImportError as exc:
            raise AdapterRegistryError(
                "Abel adapter is unavailable. See: abel_edge/plugins/AGENTS.md"
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
            end=guarded_end,
            limit=request.limit,
            required_columns=ABEL_BAR_CACHE_COLUMNS if request.kind == "bars" else None,
            max_cache_age_seconds=_max_cache_age_seconds(request.options),
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
            if effective_limit is None:
                effective_limit = 5000
            try:
                bars = fetch_bars(
                    symbols=[symbol],
                    start=request.start,
                    end=guarded_end,
                    timeframe=request.timeframe or "1d",
                    limit=effective_limit,
                    fields=ABEL_BAR_FIELDS if request.kind == "bars" else fields,
                    config=request.options,
                )
                write_cached_bars(
                    entry,
                    bars,
                    requested_start=request.start,
                    requested_end=guarded_end,
                    requested_limit=request.limit,
                )
            except missing_api_key_error as exc:
                raise AdapterRegistryError(str(exc)) from exc
        assert_frame_respects_max_data_date(bars, source=f"feed '{request.feed_name}'")

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


def point_in_time_cache_identity(
    *,
    series_spec_sha256: str,
    source_identity: str,
) -> str:
    """Hash one series spec and resolved source endpoint into a cache identity."""

    payload = {
        "series_spec_sha256": str(series_spec_sha256),
        "source_identity": str(source_identity).rstrip("/"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _point_in_time_cache_end_has_elapsed(
    end: object | None,
    *,
    now: pd.Timestamp | None = None,
) -> bool:
    if end is None:
        return False
    effective_end = pd.to_datetime(end, utc=True)
    if is_date_only_bound(end):
        effective_end += pd.Timedelta(days=1)
    current = pd.Timestamp.now(tz="UTC") if now is None else now
    return effective_end <= current


def _max_cache_age_seconds(options: dict[str, object]) -> float | None:
    raw = options.get("max_cache_age_seconds")
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None
