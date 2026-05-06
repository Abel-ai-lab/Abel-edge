"""Regression tests for the dashboard portfolio aggregators.

Locks the SSOT invariant: when ``primary_portfolio_id`` is configured,
both ``build_recent_days`` and ``build_ledger`` must use the primary
portfolio's own ``pnl`` series (already weighted across components by
the composite engine) rather than naively summing raw daily PnL%
across every registered strategy. The naive sum has no financial
meaning (each strategy's PnL% is normalized by its own capital base)
and previously caused the Recent Days widget to show ±5-15% daily
swings while the Hero panel reported MaxDD of -0.65%.
"""
from __future__ import annotations

import pandas as pd

from causal_edge.dashboard.portfolio import build_ledger, build_recent_days


def _df(rows):
    return pd.DataFrame(rows)


def _trade_log_factory(logs):
    """Returns a load_trade_log callable that maps path → DataFrame."""
    def load(path):
        return logs.get(path)
    return load


def test_recent_days_uses_primary_pnl_when_configured():
    primary = _df([
        {"date": "2026-04-30", "pnl": 0.003, "position": 0.4},
        {"date": "2026-05-01", "pnl": 0.021, "position": 0.5},
    ])
    other = _df([
        {"date": "2026-04-30", "pnl": -0.066, "position": 1.0},
        {"date": "2026-05-01", "pnl": 0.158, "position": 1.0},
    ])
    component = _df([
        {"date": "2026-04-30", "pnl": 0.001, "position": 0.5},
        {"date": "2026-05-01", "pnl": 0.005, "position": 0.5},
    ])
    logs = {
        "primary.csv": primary,
        "other.csv": other,
        "comp.csv": component,
    }
    strat_cfgs = [
        {"id": "abel_real_money", "trade_log": "primary.csv",
         "components": ["comp"]},
        {"id": "other", "trade_log": "other.csv"},
        {"id": "comp", "trade_log": "comp.csv"},
    ]
    strategies = [
        {"id": "abel_real_money", "has_data": True},
        {"id": "other", "has_data": True},
        {"id": "comp", "has_data": True},
    ]

    recent, history = build_recent_days(
        strategies, strat_cfgs, _trade_log_factory(logs),
        primary_portfolio_id="abel_real_money",
    )
    by_date = {r["date"]: r for r in recent}
    assert by_date["05-01"]["pnl"] == 0.021, (
        "Recent Days must use primary's daily pnl, not raw sum"
    )
    assert by_date["04-30"]["pnl"] == 0.003
    # n_active counts active *components* of the primary, not all strategies.
    assert by_date["05-01"]["n_active"] == 1
    assert history == [0.021, 0.003] or history == [0.003, 0.021]


def test_recent_days_falls_back_to_legacy_sum_without_primary():
    a = _df([{"date": "2026-05-01", "pnl": 0.05, "position": 1.0}])
    b = _df([{"date": "2026-05-01", "pnl": 0.10, "position": 1.0}])
    logs = {"a.csv": a, "b.csv": b}
    strat_cfgs = [
        {"id": "a", "trade_log": "a.csv"},
        {"id": "b", "trade_log": "b.csv"},
    ]
    strategies = [
        {"id": "a", "has_data": True},
        {"id": "b", "has_data": True},
    ]

    recent, _ = build_recent_days(
        strategies, strat_cfgs, _trade_log_factory(logs),
        primary_portfolio_id=None,
    )
    assert recent[0]["pnl"] == 0.05 + 0.10
    assert recent[0]["n_active"] == 2


def test_ledger_header_uses_primary_pnl_when_configured():
    primary = _df([
        {"date": "2026-05-01", "pnl": 0.021, "position": 0.5,
         "source": "backfill"},
    ])
    other = _df([
        {"date": "2026-05-01", "pnl": 0.158, "position": 1.0,
         "source": "backfill"},
    ])
    logs = {"primary.csv": primary, "other.csv": other}
    strat_cfgs = [
        {"id": "abel_real_money", "trade_log": "primary.csv"},
        {"id": "other", "trade_log": "other.csv"},
    ]
    strategies = [
        {"id": "abel_real_money", "has_data": True, "name": "Real",
         "color": "#000"},
        {"id": "other", "has_data": True, "name": "Other", "color": "#fff"},
    ]

    ledger = build_ledger(
        strategies, strat_cfgs, _trade_log_factory(logs),
        since_date="2026-05-01",
        primary_portfolio_id="abel_real_money",
    )
    rows_by_date = {r["date"]: r for r in ledger}
    assert "2026-05-01" in rows_by_date
    assert rows_by_date["2026-05-01"]["total_pnl"] == 0.021, (
        "Ledger header must use primary's pnl, not 0.021 + 0.158 = 0.179"
    )
