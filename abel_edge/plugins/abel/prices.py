"""Abel price data helpers."""

from __future__ import annotations

import pandas as pd

from abel_edge.engine.feed_contract import (
    apply_max_data_date_guard,
    assert_frame_respects_max_data_date,
)
from abel_edge.plugins.abel.client import AbelClient
from abel_edge.plugins.abel.credentials import MissingAbelApiKeyError, require_api_key

DEFAULT_BAR_FIELDS = ("open", "high", "low", "close", "volume")


def fetch_bars(
    *,
    symbols: list[str],
    start=None,
    end=None,
    timeframe: str = "1d",
    limit: int | None = None,
    fields: list[str] | None = None,
    config: dict | None = None,
    client: AbelClient | None = None,
) -> pd.DataFrame:
    env_path = (config or {}).get("env_path", ".env")
    try:
        api_key = require_api_key(env_path=env_path)
    except MissingAbelApiKeyError as e:
        raise MissingAbelApiKeyError(
            f"{e} Or set price_data.adapter to 'csv' for local bar data."
        ) from e
    guarded_end = apply_max_data_date_guard(end, source="Abel price fetch")
    abel = client or AbelClient(env_path=env_path)
    payload = abel.fetch_bars(
        symbols=symbols,
        start=start,
        end=guarded_end,
        timeframe=timeframe,
        limit=limit,
        fields=fields,
        api_key=api_key,
    )
    frame = pd.DataFrame(payload)
    if frame.empty and "timestamp" not in frame.columns:
        frame = empty_bar_frame(fields=fields)
    assert_frame_respects_max_data_date(frame, source="Abel price fetch")
    return frame


def fetch_node_series(
    *,
    node_id: str,
    start=None,
    end=None,
    limit: int | None = None,
    config: dict | None = None,
    client: AbelClient | None = None,
) -> pd.DataFrame:
    """Fetch one CAP scalar graph-node series without price adjustment."""

    env_path = (config or {}).get("env_path", ".env")
    try:
        api_key = require_api_key(env_path=env_path)
    except MissingAbelApiKeyError as e:
        raise MissingAbelApiKeyError(
            f"{e} Canonical Abel nodes require live node_id data access."
        ) from e
    guarded_end = apply_max_data_date_guard(end, source="Abel node-series fetch")
    abel = client or AbelClient(env_path=env_path)
    payload = abel.fetch_node_series(
        node_id=node_id,
        start=start,
        end=guarded_end,
        limit=limit,
        api_key=api_key,
    )
    frame = pd.DataFrame(payload)
    if frame.empty and "timestamp" not in frame.columns:
        frame = pd.DataFrame(columns=["timestamp", "node_id", "value"])
    # CAP bounds node-series responses by observation date. Their availability
    # timestamp may legitimately be later and is filtered by the PITS
    # materializer before the general frame guard runs.
    return frame


def empty_bar_frame(*, fields: list[str] | None = None) -> pd.DataFrame:
    columns = ["timestamp", "symbol"]
    for field in fields or list(DEFAULT_BAR_FIELDS):
        if field not in columns:
            columns.append(str(field))
    if "close" not in columns:
        columns.append("close")
    return pd.DataFrame(columns=columns)
