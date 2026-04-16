"""Framework-managed feed loading helpers for strategy engines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_edge.engine.feed_contract import FeedContractError, normalize_series_frame
from causal_edge.engine.price_data import CSV_ALIASES, normalize_bars


def load_declared_feed(engine, name: str, **kwargs) -> pd.DataFrame:
    feeds = (engine.context or {}).get("_feeds") or {}
    if name not in feeds:
        raise FeedContractError(f"Feed '{name}' is not declared for this strategy.")

    feed_cfg = dict(feeds[name])
    kind = feed_cfg.get("kind")
    if kind == "bars":
        return _load_bars_feed(engine, feed_cfg, **kwargs)
    if kind == "series":
        return _load_series_feed(engine, feed_cfg, **kwargs)
    raise FeedContractError(f"Unsupported feed kind '{kind}' for feed '{name}'.")


def _load_bars_feed(engine, feed_cfg: dict, **kwargs) -> pd.DataFrame:
    if engine._bars_loader is None:
        raise RuntimeError("Price loader is not configured for this engine.")
    symbols = [feed_cfg["symbol"]] if feed_cfg.get("symbol") else None
    df = engine._bars_loader(
        symbols=symbols,
        start=kwargs.get("start"),
        end=kwargs.get("end"),
        timeframe=kwargs.get("timeframe") or feed_cfg.get("timeframe", "1d"),
        limit=kwargs.get("limit"),
        fields=kwargs.get("fields"),
        config=feed_cfg,
    )
    return normalize_bars(df)


def _load_series_feed(engine, feed_cfg: dict, **kwargs) -> pd.DataFrame:
    source = feed_cfg.get("source")
    if source == "csv" and feed_cfg.get("path"):
        return _load_series_from_csv(feed_cfg, **kwargs)

    field = feed_cfg["field"]
    bars = _load_bars_feed(engine, feed_cfg, fields=[field], **kwargs)
    if field not in bars.columns:
        raise FeedContractError(
            f"Feed '{feed_cfg['name']}' could not resolve required field '{field}'."
        )
    series_df = bars[["timestamp", field]].copy()
    if "symbol" in bars.columns:
        series_df["symbol"] = bars["symbol"].astype(str)
    return normalize_series_frame(
        series_df.rename(columns={field: "value"}),
        field="value",
        name=f"feed '{feed_cfg['name']}'",
        profile=feed_cfg.get("profile", "daily"),
    )


def _load_series_from_csv(feed_cfg: dict, **kwargs) -> pd.DataFrame:
    path = Path(feed_cfg["path"])
    df = pd.read_csv(path).rename(columns=CSV_ALIASES)
    field = feed_cfg["field"]
    if field not in df.columns:
        raise FeedContractError(
            f"CSV series feed '{feed_cfg['name']}' is missing declared field '{field}'."
        )
    if "symbol" not in df.columns and feed_cfg.get("symbol"):
        df = df.copy()
        df["symbol"] = feed_cfg["symbol"]

    frame = normalize_series_frame(
        df.rename(columns={field: "value"}),
        field="value",
        name=f"feed '{feed_cfg['name']}'",
        profile=feed_cfg.get("profile", "daily"),
    )
    start = kwargs.get("start")
    end = kwargs.get("end")
    limit = kwargs.get("limit")
    if start is not None:
        frame = frame[frame["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        frame = frame[frame["timestamp"] <= pd.to_datetime(end, utc=True)]
    if limit:
        group_cols = ["symbol"] if "symbol" in frame.columns else None
        if group_cols:
            frame = frame.groupby(group_cols, group_keys=False).tail(limit)
        else:
            frame = frame.tail(limit)
    return frame.reset_index(drop=True)
