"""Minimal deterministic backtest kernel for close-to-close execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BacktestSettings:
    """Execution settings applied uniformly to a position series."""

    cost_bps: float = 0.0
    max_abs_position: float | None = None


def run_backtest(
    positions: np.ndarray,
    prices: np.ndarray,
    settings: BacktestSettings | None = None,
) -> dict[str, np.ndarray]:
    """Compute effective positions, returns, and net PnL from raw strategy outputs."""
    raw_positions = np.asarray(positions, dtype=float)
    close_prices = np.asarray(prices, dtype=float)
    if raw_positions.shape != close_prices.shape:
        raise ValueError("positions and prices must have the same shape.")

    cfg = settings or BacktestSettings()
    effective_positions = raw_positions.copy()
    if cfg.max_abs_position is not None:
        effective_positions = np.clip(
            effective_positions,
            -float(cfg.max_abs_position),
            float(cfg.max_abs_position),
        )

    asset_returns = np.zeros_like(close_prices, dtype=float)
    if len(close_prices) > 1:
        asset_returns[1:] = close_prices[1:] / close_prices[:-1] - 1.0

    turnover = np.zeros_like(effective_positions, dtype=float)
    if len(effective_positions) > 1:
        turnover[1:] = np.abs(effective_positions[1:] - effective_positions[:-1])

    execution_cost = turnover * (float(cfg.cost_bps) / 10000.0)
    gross_pnl = effective_positions * asset_returns
    net_pnl = gross_pnl - execution_cost

    if len(net_pnl) > 0:
        gross_pnl[0] = 0.0
        net_pnl[0] = 0.0
        turnover[0] = 0.0
        execution_cost[0] = 0.0

    return {
        "positions": effective_positions,
        "asset_returns": asset_returns,
        "turnover": turnover,
        "execution_cost": execution_cost,
        "gross_pnl": gross_pnl,
        "pnl": net_pnl,
    }
