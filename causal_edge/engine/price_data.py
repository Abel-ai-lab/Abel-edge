"""Price data loading helpers for strategy engines."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

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
        raise ValueError(f"Price data missing required columns: {missing}")

    renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], utc=True)
    renamed["symbol"] = renamed["symbol"].astype(str)
    renamed["close"] = renamed["close"].astype(float)
    renamed = renamed.sort_values(["symbol", "timestamp"]).drop_duplicates(
        subset=["symbol", "timestamp"], keep="last"
    )
    return renamed.reset_index(drop=True)


def load_bars_from_csv(
    path: str | Path,
    *,
    symbols: list[str],
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
    if limit:
        filtered = filtered.groupby("symbol", group_keys=False).tail(limit)
    return filtered.reset_index(drop=True)
