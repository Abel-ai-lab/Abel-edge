"""Price data loading helpers for strategy engines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from causal_edge.engine.feed_contract import FeedNormalizationError, validate_datetime_index

BarsLoader = Callable[..., pd.DataFrame]
REQUIRED_BAR_COLUMNS = ("timestamp", "symbol", "close")
CSV_ALIASES = {"date": "timestamp", "price": "close"}


def resolve_price_config(settings: dict, strategy_cfg: dict) -> dict:
    default_cfg = settings.get("price_data") or {}
    strategy_price_cfg = strategy_cfg.get("price_data") or {}
    merged = {**default_cfg, **strategy_price_cfg}
    merged.setdefault("source", default_cfg.get("default_source", "abel"))
    merged.setdefault("timeframe", merged.get("default_timeframe", "1d"))
    merged.setdefault("symbol", strategy_cfg.get("asset"))
    return merged


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(columns=CSV_ALIASES).copy()
    missing = [col for col in REQUIRED_BAR_COLUMNS if col not in renamed.columns]
    if missing:
        raise FeedNormalizationError(f"Price data missing required columns: {missing}")

    try:
        renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], utc=False)
    except (TypeError, ValueError) as exc:
        raise FeedNormalizationError("Price data contains invalid timestamp values.") from exc
    renamed["symbol"] = renamed["symbol"].astype(str)
    numeric_cols = [col for col in ["close", "open", "high", "low", "volume"] if col in renamed.columns]
    for col in numeric_cols:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
        if renamed[col].isna().any():
            raise FeedNormalizationError(f"Price data column '{col}' contains non-numeric values.")

    renamed = renamed.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    for symbol, group in renamed.groupby("symbol", sort=False):
        renamed.loc[group.index, "timestamp"] = validate_datetime_index(
            group["timestamp"],
            profile="daily",
            name=f"price_data[{symbol}].timestamp",
        )
    return renamed.reset_index(drop=True)


def load_bars_from_csv(
    path: str | Path,
    *,
    symbols: list[str],
    start=None,
    end=None,
    limit: int | None = None,
    **_: object,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        if len(symbols) != 1:
            raise ValueError("CSV price data must include 'symbol' when loading multiple symbols.")
        df = df.copy()
        df["symbol"] = symbols[0]

    bars = normalize_bars(df)
    filtered = bars[bars["symbol"].isin(symbols)]
    if start is not None:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        filtered = filtered[filtered["timestamp"] <= pd.to_datetime(end, utc=True)]
    if limit:
        filtered = filtered.groupby("symbol", group_keys=False).tail(limit)
    return filtered.reset_index(drop=True)
