"""Helpers for dashboard price overlays sourced from external bars."""

from __future__ import annotations

import pandas as pd

from causal_edge.engine.price_data import resolve_price_config
from causal_edge.engine.trader import _load_engine


def _fetch_price_overlay_from_engine(s_cfg: dict, settings: dict, bars_loader) -> dict | None:
    try:
        engine_cls = _load_engine(s_cfg["engine"])
        engine = engine_cls(context=s_cfg)
        if bars_loader is not None:
            engine.bind_price_loader(bars_loader, resolve_price_config(settings or {}, s_cfg))
        signal_data = engine.compute_signals()
    except Exception:
        return None

    if not isinstance(signal_data, tuple):
        return None

    if len(signal_data) == 3:
        _, dates, prices = signal_data
    elif len(signal_data) == 4:
        _, dates, _, prices = signal_data
    else:
        return None

    if dates is None or prices is None or len(dates) == 0 or len(prices) == 0:
        return None

    dates = pd.DatetimeIndex(dates)
    closes = pd.Series(prices, index=dates, dtype=float)
    closes = closes.sort_index().drop_duplicates(keep="last")
    returns = closes.pct_change().fillna(0.0)
    return {
        "dates": pd.DatetimeIndex(closes.index),
        "returns": returns.values.astype(float),
        "close": closes.values.astype(float),
    }


def fetch_price_overlay(s_cfg: dict, settings: dict, bars_loader, *, start, end) -> dict | None:
    if bars_loader is None:
        return None
    if str(s_cfg.get("asset") or "").upper() == "MULTI":
        return None

    price_config = resolve_price_config(settings or {}, s_cfg)
    symbol = str(price_config.get("symbol") or s_cfg["asset"])
    if bars_loader is not None:
        try:
            bars = bars_loader(
                symbols=[symbol],
                start=start,
                end=end,
                timeframe=price_config.get("timeframe", "1d"),
                fields=["close"],
                config=price_config,
            )
        except Exception:
            bars = None

        if bars is not None and len(bars) > 0:
            filtered = bars[bars["symbol"].astype(str) == symbol].copy()
            if len(filtered) == 0:
                filtered = bars.copy()
            filtered = filtered.sort_values("timestamp").drop_duplicates(
                subset=["timestamp"], keep="last"
            )
            if len(filtered) > 0:
                closes = filtered["close"].astype(float)
                returns = closes.pct_change().fillna(0.0)
                return {
                    "dates": pd.DatetimeIndex(filtered["timestamp"]),
                    "returns": returns.values.astype(float),
                    "close": closes.values.astype(float),
                }

    overlay = _fetch_price_overlay_from_engine(s_cfg, settings, bars_loader)
    if overlay is None:
        return None

    dates = overlay["dates"]
    mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
    if mask.sum() == 0:
        return None
    return {
        "dates": dates[mask],
        "returns": overlay["returns"][mask],
        "close": overlay["close"][mask],
    }
