"""Stateless chart-builder functions for the dashboard.

Every public function: data in (arrays/dicts) → string out (JSON or HTML).
No side effects. No strategy-specific logic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - depends on optional dependency
    go = None


def _empty_chart_json() -> str:
    """Return a minimal empty chart payload when Plotly is unavailable."""
    return json.dumps({"data": [], "layout": {}})


def _chart_to_json(fig) -> str:
    """Convert Plotly figure to JSON string safe for inline <script> embedding."""
    raw = json.dumps(fig.to_dict(), default=str)
    return raw.replace("</", r"<\/")


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color to an rgba() string with the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def compute_metrics(pnl: np.ndarray, periods_per_year: int = 252) -> dict:
    """Compute standard metrics from a PnL array of daily simple returns.

    Returns dict with: sharpe, cum_return, max_dd, win_rate, n_trades, n_days.
    """
    if len(pnl) == 0:
        return dict(sharpe=0, cum_return=0, max_dd=0, win_rate=0, n_trades=0, n_days=0)

    equity = np.cumprod(1.0 + pnl)
    cum_return = equity - 1.0
    std = np.std(pnl, ddof=1) if len(pnl) > 1 else 0.0
    sharpe = float(np.mean(pnl) / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    active = pnl[np.abs(pnl) > 1e-10]
    win_rate = float(np.mean(active > 0)) if len(active) > 0 else 0.0
    n_trades = int(np.sum(np.abs(pnl) > 1e-10))

    return dict(
        sharpe=round(sharpe, 2),
        cum_return=round(float(cum_return[-1]), 4),
        max_dd=round(abs(max_dd), 4),
        win_rate=round(win_rate, 3),
        n_trades=n_trades,
        n_days=len(pnl),
    )


def equity_chart(
    dates, cum_return, name: str, color: str, asset_index=None, asset: str | None = None
) -> str:
    """Equity curve chart, optionally overlaid with underlying asset trend."""
    if go is None:
        return _empty_chart_json()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=list(cum_return),
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.12),
        )
    )
    if asset_index is not None and asset is not None:
        fig.add_trace(
            go.Scatter(
                x=list(dates),
                y=list(np.asarray(asset_index, dtype=float) - 1.0),
                mode="lines",
                name=f"{asset} Price",
                line=dict(color="#94A3B8", width=2, dash="dash"),
            )
        )
    fig.update_layout(
        title=f"{name} — Backtest vs {asset or 'Asset'}",
        xaxis_title="Date",
        yaxis_title="Normalized Return",
        yaxis_tickformat=".1%",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return _chart_to_json(fig)


def asset_price_chart(dates, asset_returns, asset: str, color: str) -> str:
    """Underlying asset normalized price chart. Returns JSON string."""
    if go is None:
        return _empty_chart_json()

    asset_index = np.cumprod(1.0 + np.asarray(asset_returns, dtype=float))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=list(asset_index),
            mode="lines",
            name=asset,
            line=dict(color=color, width=2),
        )
    )
    fig.update_layout(
        title=f"{asset} — Price Trend",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return _chart_to_json(fig)


def asset_index_from_returns(asset_returns) -> np.ndarray:
    """Build normalized asset index from simple returns."""
    return np.cumprod(1.0 + np.asarray(asset_returns, dtype=float))


def position_chart(dates, positions, name: str, color: str) -> str:
    """Position history chart. Returns JSON string.

    Args:
        dates: array-like of dates
        positions: array-like of position sizes (0=flat, 1=long)
        name: strategy name for legend
        color: hex color string
    """
    if go is None:
        return _empty_chart_json()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=list(positions),
            mode="lines",
            name="Position",
            line=dict(color=color, width=1),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.19),
        )
    )
    fig.update_layout(
        title=f"{name} — Position",
        xaxis_title="Date",
        yaxis_title="Position Size",
        template="plotly_dark",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return _chart_to_json(fig)
