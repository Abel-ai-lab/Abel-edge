"""Framework-managed feed loading helpers for strategy engines."""

from __future__ import annotations

from typing import Any

import pandas as pd

from causal_edge.engine.adapter_registry import FeedLoadRequest, resolve_adapter
from causal_edge.engine.feed_contract import FeedContractError, normalize_series_frame
from causal_edge.engine.price_data import normalize_bars

_REQUEST_RESERVED_KEYS = {
    "adapter",
    "kind",
    "symbol",
    "field",
    "timeframe",
    "profile",
    "name",
}


def load_declared_feed(engine, name: str, **kwargs) -> pd.DataFrame:
    feeds = (engine.context or {}).get("_feeds") or {}
    if name not in feeds:
        raise FeedContractError(f"Feed '{name}' is not declared for this strategy.")
    return load_feed_frame(
        feeds[name],
        strategy_id=(engine.context or {}).get("id"),
        **kwargs,
    )


def load_feed_frame(
    feed_cfg: dict[str, Any],
    *,
    strategy_id: str | None = None,
    start=None,
    end=None,
    timeframe: str | None = None,
    limit: int | None = None,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    feed_name = str(feed_cfg.get("name") or "feed")
    kind = str(feed_cfg.get("kind") or "").strip().lower()
    if kind not in {"bars", "series"}:
        raise FeedContractError(f"Unsupported feed kind '{kind}' for feed '{feed_name}'.")

    adapter_name = str(feed_cfg.get("adapter") or "").strip().lower()
    if not adapter_name:
        raise FeedContractError(f"Feed '{feed_name}' is missing required adapter configuration.")

    adapter = resolve_adapter(adapter_name)
    request_fields = _request_fields(kind, fields)
    request = FeedLoadRequest(
        adapter=adapter_name,
        kind=kind,
        symbol=feed_cfg.get("symbol"),
        field=feed_cfg.get("field"),
        timeframe=timeframe or feed_cfg.get("timeframe"),
        start=start,
        end=end,
        limit=limit,
        profile=str(feed_cfg.get("profile") or "daily"),
        options=_request_options(feed_cfg, fields=request_fields),
        strategy_id=strategy_id,
        feed_name=feed_name,
    )
    raw = adapter.load(request)
    frame = _normalize_loaded_frame(feed_cfg, raw, assume_utc_for_naive=adapter.assume_utc_for_naive)
    return _apply_time_filters(frame, start=start, end=end, limit=limit)


def _normalize_loaded_frame(
    feed_cfg: dict[str, Any],
    df: pd.DataFrame,
    *,
    assume_utc_for_naive: bool,
) -> pd.DataFrame:
    kind = feed_cfg["kind"]
    name = f"feed '{feed_cfg['name']}'"
    if kind == "bars":
        frame = normalize_bars(df, assume_utc_for_naive=assume_utc_for_naive)
        symbol = feed_cfg.get("symbol")
        if symbol:
            frame = frame[frame["symbol"].astype(str) == str(symbol)].copy()
        return frame.reset_index(drop=True)
    if kind == "series":
        frame = df.copy()
        if "symbol" not in frame.columns and feed_cfg.get("symbol"):
            frame["symbol"] = str(feed_cfg["symbol"])
        return normalize_series_frame(
            frame,
            field="value",
            name=name,
            profile=feed_cfg.get("profile", "daily"),
            assume_utc_for_naive=assume_utc_for_naive,
        )
    raise FeedContractError(f"Unsupported feed kind '{kind}' for feed '{feed_cfg['name']}'.")


def _apply_time_filters(frame: pd.DataFrame, *, start=None, end=None, limit: int | None = None):
    filtered = frame
    if start is not None:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        filtered = filtered[filtered["timestamp"] <= pd.to_datetime(end, utc=True)]
    if limit:
        group_cols = ["symbol"] if "symbol" in filtered.columns else None
        if group_cols:
            filtered = filtered.groupby(group_cols, group_keys=False).tail(limit)
        else:
            filtered = filtered.tail(limit)
    return filtered.reset_index(drop=True)


def _request_fields(kind: str, fields: list[str] | None) -> list[str] | None:
    if kind != "bars":
        return None
    if not fields:
        return None
    requested = [str(field) for field in fields]
    if "close" not in requested:
        requested.append("close")
    return requested


def _request_options(feed_cfg: dict[str, Any], *, fields: list[str] | None) -> dict[str, object]:
    options = {
        key: value for key, value in feed_cfg.items() if key not in _REQUEST_RESERVED_KEYS and value is not None
    }
    if fields is not None:
        options["fields"] = fields
    return options
