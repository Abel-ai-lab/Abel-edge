"""Minimal DecisionContext SMA crossover example."""

from __future__ import annotations

from abel_edge.engine.base import StrategyEngine


class SMAEngine(StrategyEngine):
    """Simple moving-average crossover on the primary target feed."""

    def __init__(self, context: dict | None = None) -> None:
        super().__init__(context=context)
        self.fast = 10
        self.slow = 30

    def compute_decisions(self, ctx):
        close = ctx.target.series("close")
        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()
        next_position = (fast_ma > slow_ma).astype(float).fillna(0.0)
        if len(next_position) > 0:
            next_position.iloc[: self.slow] = 0.0
        return ctx.decisions(next_position)
