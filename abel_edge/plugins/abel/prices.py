"""Abel price data helpers."""

from __future__ import annotations

import pandas as pd

from abel_edge.plugins.abel.client import AbelClient
from abel_edge.plugins.abel.credentials import MissingAbelApiKeyError, require_api_key


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
    abel = client or AbelClient()
    payload = abel.fetch_bars(
        symbols=symbols,
        start=start,
        end=end,
        timeframe=timeframe,
        limit=limit,
        fields=fields,
        api_key=api_key,
    )
    return pd.DataFrame(payload)
