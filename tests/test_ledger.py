"""Regression tests for trade-log uniqueness and runtime leak wiring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from abel_edge.engine.ledger import append_trade_log_rows, read_trade_log, write_trade_log
from abel_edge.validation.gate import validate_strategy


def test_append_trade_rows_prefer_live_over_same_day_backfill(tmp_path) -> None:
    path = tmp_path / "log.csv"
    dates = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True)
    write_trade_log(
        dates,
        np.array([0.0, 0.10]),
        np.array([0.0, 0.10]),
        np.array([0.0, 1.0]),
        path,
        close_prices=np.array([100.0, 110.0]),
    )

    append_trade_log_rows(
        path,
        [
            {
                "date": dates[-1],
                "close": 111.0,
                "asset_return": 0.05,
                "position": 1.0,
                "pnl": 0.05,
                "next_position": 0.0,
                "source": "live",
            }
        ],
    )

    df = read_trade_log(path)
    assert len(df) == 2
    assert df["date"].duplicated().sum() == 0
    assert df.iloc[-1]["source"] == "live"
    assert float(df.iloc[-1]["pnl"]) == 0.05


def test_write_trade_log_cum_return_uses_simple_interest(tmp_path) -> None:
    path = tmp_path / "log.csv"
    dates = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True)
    write_trade_log(
        dates,
        np.array([0.10, 0.10]),
        np.array([0.10, 0.10]),
        np.array([1.0, 1.0]),
        path,
    )

    df = read_trade_log(path)
    assert float(df.iloc[-1]["cum_return"]) == 0.20


def test_write_trade_log_preserves_existing_live_rows(tmp_path) -> None:
    path = tmp_path / "log.csv"
    dates = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True)
    write_trade_log(
        dates,
        np.array([0.0, 0.10]),
        np.array([0.0, 0.10]),
        np.array([0.0, 1.0]),
        path,
    )
    append_trade_log_rows(
        path,
        [
            {
                "date": dates[-1],
                "asset_return": 0.05,
                "position": 1.0,
                "pnl": 0.05,
                "source": "live",
            }
        ],
    )

    new_dates = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
        utc=True,
    )
    write_trade_log(
        new_dates,
        np.array([0.0, 0.10, -0.02]),
        np.array([0.0, 0.10, -0.02]),
        np.array([0.0, 1.0, 0.0]),
        path,
    )

    df = read_trade_log(path)
    assert len(df) == 3
    assert df["date"].duplicated().sum() == 0
    same_day = df[df["date"] == dates[-1]]
    assert len(same_day) == 1
    assert same_day.iloc[0]["source"] == "live"


def test_append_trade_log_rows_preserves_unknown_columns(tmp_path) -> None:
    path = tmp_path / "log.csv"
    append_trade_log_rows(
        path,
        [
            {
                "date": pd.Timestamp("2026-01-01", tz="UTC"),
                "asset_return": 0.0,
                "position": 0.0,
                "pnl": 0.0,
                "source": "live",
                "paper_audit_status": "provider_fetch",
            }
        ],
    )

    df = read_trade_log(path)
    assert df.iloc[0]["paper_audit_status"] == "provider_fetch"


def test_write_trade_log_preserves_live_unknown_columns(tmp_path) -> None:
    path = tmp_path / "log.csv"
    dates = pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)
    write_trade_log(dates, np.array([0.0]), np.array([0.0]), np.array([0.0]), path)
    append_trade_log_rows(
        path,
        [
            {
                "date": dates[0],
                "asset_return": 0.0,
                "position": 0.0,
                "pnl": 0.0,
                "source": "live",
                "paper_audit_status": "provider_fetch",
            }
        ],
    )
    write_trade_log(dates, np.array([0.0]), np.array([0.0]), np.array([0.0]), path)

    df = read_trade_log(path)
    assert df.iloc[0]["source"] == "live"
    assert df.iloc[0]["paper_audit_status"] == "provider_fetch"


def test_validate_strategy_surfaces_runtime_r1_failure(tmp_path) -> None:
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n)
    asset_returns = rng.normal(0, 0.01, n)
    positions = np.abs(asset_returns) * 100
    pnl = positions * asset_returns

    path = tmp_path / "leaky.csv"
    pd.DataFrame(
        {
            "date": dates,
            "asset_return": asset_returns,
            "pnl": pnl,
            "position": positions,
            "cum_return": np.cumsum(pnl),
            "source": ["backfill"] * n,
        }
    ).to_csv(path, index=False)

    result = validate_strategy(path, profile="equity_daily")
    assert result["verdict"] == "FAIL"
    assert any(message.startswith("R1") for message in result["failures"])
