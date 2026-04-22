"""Trade log read/write. Single source of truth for trade log CSV format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "pnl", "position", "cum_return", "source")


def read_trade_log(path: str | Path) -> pd.DataFrame:
    """Read a trade log CSV. Returns DataFrame with standard columns.

    Date parsing uses `format="mixed"` because backfill rows write midnight
    timestamps ("2026-04-16 00:00:00+00:00") while live rows carry
    sub-second ISO-8601 ("2026-04-17T05:55:06.150276+00:00"). The default
    strptime fallback on mixed formats raises, which blocked every
    subsequent `causal-edge run` once any live row existed.
    """
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    if "decision_time" in df.columns:
        df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, format="mixed")
    if "effective_time" in df.columns:
        df["effective_time"] = pd.to_datetime(df["effective_time"], utc=True, format="mixed")
    return df


def write_trade_log(
    dates: pd.DatetimeIndex,
    asset_returns: np.ndarray,
    pnl: np.ndarray,
    positions: np.ndarray,
    path: str | Path,
    source: str = "backfill",
    decision_times: pd.DatetimeIndex | None = None,
    effective_times: pd.DatetimeIndex | None = None,
    close_prices: np.ndarray | None = None,
    next_positions: np.ndarray | None = None,
    gross_pnl: np.ndarray | None = None,
    turnover: np.ndarray | None = None,
    execution_cost: np.ndarray | None = None,
) -> None:
    """Write a trade log CSV from strategy output arrays.

    Args:
        dates: Trading dates
        asset_returns: Daily simple returns of the underlying asset
        pnl: Daily net PnL after execution costs
        positions: Daily position sizes
        path: Output CSV path
        source: "backfill" or "live"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "date": dates,
            "asset_return": asset_returns,
            "pnl": pnl,
            "position": positions,
            "source": source,
        }
    )
    if decision_times is not None:
        df["decision_time"] = decision_times
    if effective_times is not None:
        df["effective_time"] = effective_times
    if close_prices is not None:
        df["close"] = close_prices
    if next_positions is not None:
        df["next_position"] = next_positions
    if gross_pnl is not None:
        df["gross_pnl"] = gross_pnl
    if turnover is not None:
        df["turnover"] = turnover
    if execution_cost is not None:
        df["execution_cost"] = execution_cost
    if path.exists():
        existing = read_trade_log(path)
        if "source" in existing.columns:
            existing["source"] = existing["source"].fillna("backfill").astype(str)
            existing["date"] = pd.to_datetime(existing["date"], utc=True)
            live_rows = existing[existing["source"].str.lower() == "live"].copy()
            if not live_rows.empty:
                live_rows = _dedupe_trade_rows(live_rows)
                df["date"] = pd.to_datetime(df["date"], utc=True)
                df = pd.concat([df, live_rows], ignore_index=True, sort=False)

    df = _dedupe_trade_rows(df)
    df["cum_return"] = np.cumprod(1.0 + df["pnl"].to_numpy(dtype=float)) - 1.0
    df.to_csv(path, index=False)


def append_trade_log_rows(path: str | Path, rows: list[dict]) -> pd.DataFrame:
    """Append live paper-trading rows and recompute cumulative return."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return read_trade_log(path) if path.exists() else incoming

    incoming["date"] = pd.to_datetime(incoming["date"], utc=True)

    if path.exists():
        existing = read_trade_log(path)
    else:
        existing = pd.DataFrame(columns=incoming.columns)

    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    combined = _dedupe_trade_rows(combined)
    combined["pnl"] = combined["pnl"].astype(float)
    combined["cum_return"] = np.cumprod(1.0 + combined["pnl"].to_numpy(dtype=float)) - 1.0
    combined.to_csv(path, index=False)
    return combined.reset_index(drop=True)


def _dedupe_trade_rows(df: pd.DataFrame) -> pd.DataFrame:
    combined = df.copy()
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], utc=True)
    if "source" not in combined.columns:
        combined["source"] = "backfill"
    combined["source"] = combined["source"].fillna("backfill").astype(str)
    combined["_src_rank"] = combined["source"].str.lower().map({"live": 1, "backfill": 0}).fillna(0)
    combined = combined.sort_values(["date", "_src_rank"], kind="mergesort")
    combined = combined.drop_duplicates(subset=["date"], keep="last")
    return combined.drop(columns="_src_rank").reset_index(drop=True)
