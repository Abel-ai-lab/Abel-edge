"""DecisionContext example showing target plus auxiliary feed reads."""

from __future__ import annotations

from causal_edge.engine.base import StrategyEngine


class FeedOverlayDemoEngine(StrategyEngine):
    """Combine a driver bars feed with a scalar risk overlay feed."""

    def compute_decisions(self, ctx):
        features = ctx.inputs_frame("btc_ref", "risk_scale")
        btc_close = features["btc_ref"].astype(float)
        risk_scale = features["risk_scale"].astype(float)

        btc_trend = (btc_close > btc_close.rolling(2, min_periods=2).mean()).astype(float)
        next_position = (risk_scale * btc_trend.fillna(0.0)).clip(lower=0.0, upper=1.0)
        return ctx.decisions(next_position)
