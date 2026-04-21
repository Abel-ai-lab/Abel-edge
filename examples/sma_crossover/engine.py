"""Minimal SMA crossover strategy engine — example implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_edge.engine.base import StrategyEngine


class SMAEngine(StrategyEngine):
    """Simple moving average crossover on synthetic price data."""

    def __init__(self, context: dict | None = None, n_days: int = 500) -> None:
        super().__init__(context=context)
        self.n_days = n_days
        self.fast = 10
        self.slow = 30

    def compute_signals(self):
        """Generate synthetic prices and compute SMA crossover signals."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.02, self.n_days)
        prices = 100.0 * np.cumprod(1.0 + returns)
        dates = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").normalize(),
            periods=self.n_days,
            freq="B",
            tz="UTC",
        )

        fast_ma = pd.Series(prices).rolling(self.fast).mean().shift(1).values
        slow_ma = pd.Series(prices).rolling(self.slow).mean().shift(1).values
        positions = np.where(fast_ma > slow_ma, 1.0, 0.0)
        positions[: self.slow + 1] = 0.0  # +1 for shift; no signal until slow MA warms up

        return self.finalize_signals(positions, dates, prices)

    def get_latest_signal(self):
        """Return latest position from the crossover."""
        positions, dates, prices = self.compute_signals()
        return {
            "position": float(positions[-1]),
            "date": str(dates[-1].date()),
            "price": float(prices[-1]),
        }
