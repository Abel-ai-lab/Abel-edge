"""Framework-managed feed loading helpers for strategy engines."""

from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any

import pandas as pd

from abel_edge.engine.adapter_registry import FeedLoadRequest, resolve_adapter
from abel_edge.engine.feed_contract import (
    FeedContractError,
    apply_max_data_date_guard,
    assert_frame_respects_max_data_date,
    normalize_series_frame,
)
from abel_edge.engine.price_data import normalize_bars
from abel_edge.engine.point_in_time_series import (
    PointInTimeSeriesSpec,
    assert_point_in_time_adapter_identity,
    normalize_point_in_time_series_frame,
)

_REQUEST_RESERVED_KEYS = {
    "adapter",
    "kind",
    "symbol",
    "field",
    "timeframe",
    "profile",
    "name",
    "series_spec",
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
    request_end=None,
    request_limit: int | None = None,
    timeframe: str | None = None,
    limit: int | None = None,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    feed_name = str(feed_cfg.get("name") or "feed")
    kind = str(feed_cfg.get("kind") or "").strip().lower()
    if kind not in {"bars", "series", "point_in_time_series"}:
        raise FeedContractError(f"Unsupported feed kind '{kind}' for feed '{feed_name}'.")

    adapter_name = str(feed_cfg.get("adapter") or "").strip().lower()
    if not adapter_name:
        raise FeedContractError(f"Feed '{feed_name}' is missing required adapter configuration.")

    adapter = resolve_adapter(adapter_name)
    series_spec = None
    if kind == "point_in_time_series":
        series_spec = PointInTimeSeriesSpec.from_mapping(feed_cfg.get("series_spec"))
        if series_spec.source_adapter != adapter_name:
            raise FeedContractError(
                f"Feed '{feed_name}' adapter '{adapter_name}' does not match "
                f"series_spec source adapter '{series_spec.source_adapter}'."
            )
    abel_node_series = bool(
        adapter_name == "abel"
        and series_spec is not None
        and series_spec.source_request.get("retrieval_mode") == "node_series"
    )
    request_fields = _request_fields(kind, fields)
    if end is not None:
        apply_max_data_date_guard(end, source=f"feed '{feed_name}' visible window")
    requested_end = end if request_end is None else request_end
    if (
        kind == "point_in_time_series"
        and not abel_node_series
        and requested_end is None
        and feed_cfg.get("source_end") is not None
    ):
        requested_end = feed_cfg["source_end"]
    guarded_request_end = apply_max_data_date_guard(
        requested_end,
        source=f"feed '{feed_name}' adapter request",
    )
    request = FeedLoadRequest(
        adapter=adapter_name,
        kind=kind,
        symbol=feed_cfg.get("symbol"),
        field=feed_cfg.get("field"),
        timeframe=timeframe or feed_cfg.get("timeframe"),
        start=start,
        end=guarded_request_end,
        limit=limit if request_limit is None else request_limit,
        profile=str(feed_cfg.get("profile") or "daily"),
        options=_request_options(feed_cfg, fields=request_fields),
        strategy_id=strategy_id,
        feed_name=feed_name,
        series_spec=series_spec,
    )
    raw = adapter.load(request)
    if series_spec is not None:
        assert_point_in_time_adapter_identity(
            raw,
            series_spec,
            name=f"feed '{feed_name}'",
            verify_source_receipt=not abel_node_series,
        )
    frame = _normalize_loaded_frame(feed_cfg, raw, assume_utc_for_naive=adapter.assume_utc_for_naive)
    assert_frame_respects_max_data_date(frame, source=f"feed '{feed_name}'")
    return _apply_time_filters(
        frame,
        start=start,
        end=end,
        limit=limit,
        date_only_end_is_inclusive=kind == "point_in_time_series",
    )


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
    if kind == "point_in_time_series":
        return normalize_point_in_time_series_frame(
            df,
            feed_cfg["series_spec"],
            name=name,
            assume_utc_for_naive=assume_utc_for_naive,
        )
    raise FeedContractError(f"Unsupported feed kind '{kind}' for feed '{feed_cfg['name']}'.")


def _apply_time_filters(
    frame: pd.DataFrame,
    *,
    start=None,
    end=None,
    limit: int | None = None,
    date_only_end_is_inclusive: bool = False,
):
    attrs = dict(frame.attrs)
    filtered = frame
    if start is not None:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        end_timestamp = pd.to_datetime(end, utc=True)
        if date_only_end_is_inclusive and _is_date_only(end):
            end_timestamp += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        filtered = filtered[filtered["timestamp"] <= end_timestamp]
    if limit:
        group_cols = ["symbol"] if "symbol" in filtered.columns else None
        if group_cols:
            filtered = filtered.groupby(group_cols, group_keys=False).tail(limit)
        else:
            filtered = filtered.tail(limit)
    filtered = filtered.reset_index(drop=True)
    filtered.attrs.update(attrs)
    return filtered


def _is_date_only(value: Any) -> bool:
    if isinstance(value, datetime):
        return False
    if isinstance(value, date):
        return True
    return isinstance(value, str) and bool(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip())
    )


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
