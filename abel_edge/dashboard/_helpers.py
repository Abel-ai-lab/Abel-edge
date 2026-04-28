"""Formatting helpers for dashboard rendering."""

from __future__ import annotations


def fmt_pnl_pct(value: float) -> str:
    """Format PnL as percentage string."""
    return f"{value * 100:+.1f}%"


def fmt_dollar(value: float) -> str:
    """Format dollar amount."""
    return f"${value:,.0f}"
