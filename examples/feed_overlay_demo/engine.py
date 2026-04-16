"""Example strategy showing the framework-managed auxiliary feed path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_edge.engine.base import StrategyEngine


class FeedOverlayDemoEngine(StrategyEngine):
    """Combine a cross-asset bars feed with a scalar risk overlay feed."""

    def compute_signals(self):
        bars = self.load_bars(limit=400)
        target = (
            bars[bars["symbol"] == self.context.get("asset", "ETHUSD")]
            .copy()
            .sort_values("timestamp")
        )
        dates = pd.DatetimeIndex(target["timestamp"])
        prices = target["close"].astype(float).to_numpy()

        btc_close = self.feed_series(
            "btc_ref",
            field="close",
            align_to=dates,
            method="ffill",
            allow_gaps=False,
        ).astype(float)
        risk_scale = self.feed_series(
            "risk_scale",
            align_to=dates,
            method="ffill",
            allow_gaps=False,
        ).astype(float)

        btc_trend = (btc_close > btc_close.rolling(2).mean().shift(1)).astype(float)
        positions = (risk_scale * btc_trend.fillna(0.0)).clip(lower=0.0, upper=1.0).to_numpy()
        return self.finalize_signals(positions, dates, prices)

    def get_latest_signal(self):
        positions, dates, prices = self.compute_signals()
        return {
            "position": float(positions[-1]),
            "date": str(dates[-1].date()),
            "price": float(prices[-1]),
        }
