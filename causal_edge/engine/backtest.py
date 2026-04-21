"""Minimal deterministic backtest kernel for close-to-close execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestSettings:
    """Execution settings applied uniformly to a position series."""

    cost_bps: float = 0.0
    max_abs_position: float | None = None


@dataclass(frozen=True)
class ExecutionFrame:
    """Kernel-owned execution artifact."""

    decision_time: pd.DatetimeIndex | None
    effective_time: pd.DatetimeIndex | None
    position: np.ndarray
    next_position: np.ndarray
    close: np.ndarray
    asset_return: np.ndarray
    turnover: np.ndarray
    execution_cost: np.ndarray
    gross_pnl: np.ndarray
    pnl: np.ndarray
    input_semantics: str


def compile_effective_positions(
    next_position: np.ndarray,
    *,
    execution_delay_bars: int,
    max_abs_position: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compile branch intent into effective positions under one timing model."""
    raw_next_position = np.asarray(next_position, dtype=float)
    if raw_next_position.ndim != 1:
        raise ValueError("next_position must be a 1D numeric array.")
    if not np.isfinite(raw_next_position).all():
        raise ValueError("next_position contains non-finite numeric values.")

    clipped_next_position = raw_next_position.copy()
    if max_abs_position is not None:
        clipped_next_position = np.clip(
            clipped_next_position,
            -float(max_abs_position),
            float(max_abs_position),
        )

    effective_positions = np.zeros_like(clipped_next_position, dtype=float)
    delay = int(execution_delay_bars)
    if delay < 0:
        raise ValueError("execution_delay_bars must be >= 0.")
    if delay == 0:
        effective_positions = clipped_next_position.copy()
    elif len(effective_positions) > delay:
        effective_positions[delay:] = clipped_next_position[:-delay]
    return effective_positions, clipped_next_position


def run_backtest(
    positions: np.ndarray,
    prices: np.ndarray,
    *,
    dates=None,
    settings: BacktestSettings | None = None,
    input_semantics: str = "effective_position",
    execution_delay_bars: int = 1,
) -> dict[str, object]:
    """Compute effective positions, returns, and net PnL from strategy outputs."""
    raw_positions = np.asarray(positions, dtype=float)
    close_prices = np.asarray(prices, dtype=float)
    if raw_positions.shape != close_prices.shape:
        raise ValueError("positions and prices must have the same shape.")

    cfg = settings or BacktestSettings()
    if input_semantics == "next_position":
        effective_positions, next_position = compile_effective_positions(
            raw_positions,
            execution_delay_bars=execution_delay_bars,
            max_abs_position=cfg.max_abs_position,
        )
    elif input_semantics == "effective_position":
        effective_positions = raw_positions.copy()
        if cfg.max_abs_position is not None:
            effective_positions = np.clip(
                effective_positions,
                -float(cfg.max_abs_position),
                float(cfg.max_abs_position),
            )
        next_position = effective_positions.copy()
    else:
        raise ValueError(
            "input_semantics must be 'effective_position' or 'next_position'."
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

    effective_time = None if dates is None else pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    decision_time = None
    if effective_time is not None:
        if input_semantics == "next_position":
            decision_time = effective_time.copy()
        else:
            decision_time = effective_time.copy()

    frame = ExecutionFrame(
        decision_time=decision_time,
        effective_time=effective_time,
        position=effective_positions,
        next_position=next_position,
        close=close_prices,
        asset_return=asset_returns,
        turnover=turnover,
        execution_cost=execution_cost,
        gross_pnl=gross_pnl,
        pnl=net_pnl,
        input_semantics=input_semantics,
    )

    return {
        "decision_time": frame.decision_time,
        "effective_time": frame.effective_time,
        "positions": effective_positions,
        "next_position": next_position,
        "asset_returns": asset_returns,
        "turnover": turnover,
        "execution_cost": execution_cost,
        "gross_pnl": gross_pnl,
        "pnl": net_pnl,
        "input_semantics": input_semantics,
        "execution_frame": frame,
    }
