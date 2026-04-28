"""ETHUSD causal strategy using Abel discovery output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from abel_edge.engine.base import StrategyEngine

GRAPH_PATH = Path(__file__).parent / "causal_graph.json"
CONVICTION_MIN = 0.75
DEFAULT_LAG = 1
DEFAULT_WINDOW = 5


class ETHUSDCausalEngine(StrategyEngine):
    """Vote-based causal strategy for ETHUSD."""

    def __init__(self, context: dict | None = None, n_days: int = 600) -> None:
        super().__init__(context=context)
        self.n_days = n_days
        with open(GRAPH_PATH, encoding="utf-8") as f:
            self.graph = json.load(f)

    def compute_signals(self):
        positions, dates, target_prices, _ = self._compute_history()
        return positions, dates, target_prices

    def get_paper_signal(self, *, as_of=None):
        positions, dates, target_prices, components = self._compute_history(as_of=as_of)
        if len(dates) == 0:
            raise ValueError("No bars available to compute paper signal.")
        next_position = self._compute_next_position(components, pd.DatetimeIndex(dates))
        return {
            "next_position": float(next_position),
            "date": str(dates[-1]),
            "price": float(target_prices[-1]),
        }

    def _compute_history(self, as_of=None):
        components = [self._normalize_component(item) for item in self.graph.get("parents", [])]
        symbols = [self.context.get("asset", "ETHUSD")] + [comp["ticker"] for comp in components]
        bars = self.load_bars(symbols=symbols, limit=self.n_days, end=as_of)

        target_symbol = symbols[0]
        target = bars[bars["symbol"] == target_symbol].copy()
        if len(target) < 30:
            raise ValueError(f"Not enough price data for {target_symbol}: need 30+ bars.")

        target = target.sort_values("timestamp").tail(self.n_days)
        target_prices = target["close"].astype(float).to_numpy()
        dates = pd.DatetimeIndex(target["timestamp"])

        sig_matrix = []
        aligned_returns = []
        for comp in components:
            tau = comp["lag"]
            win = comp["window"]
            comp_bars = bars[bars["symbol"] == comp["ticker"]].copy()
            comp_bars = comp_bars.sort_values("timestamp")
            aligned = comp_bars.set_index("timestamp")["close"].reindex(dates).ffill()
            ret = aligned.pct_change().fillna(0.0)
            aligned_returns.append(ret)
            if win > 1:
                sig = np.sign(ret.rolling(win).sum().shift(tau)).values
            else:
                sig = np.sign(ret.shift(tau)).values
            sig_matrix.append(np.nan_to_num(sig, nan=0.0))

        sig_matrix = np.array(sig_matrix)
        n_days = len(dates)
        n_up = (sig_matrix > 0).sum(axis=0)
        n_down = (sig_matrix < 0).sum(axis=0)
        n_active = (sig_matrix != 0).sum(axis=0)
        vote_frac = np.divide(
            n_up,
            n_active,
            out=np.full(n_days, 0.5, dtype=float),
            where=n_active > 0,
        )

        positions = np.zeros(n_days)
        bull = n_up > n_down
        positions[bull] = vote_frac[bull] ** 2
        positions[bull & (vote_frac < CONVICTION_MIN)] = 0.0
        return (
            np.maximum(positions, 0.0),
            dates,
            target_prices,
            list(zip(components, aligned_returns)),
        )

    def get_latest_signal(self):
        positions, dates, prices = self.compute_signals()
        return {
            "position": float(positions[-1]),
            "date": str(dates[-1].date()),
            "price": float(prices[-1]),
        }

    def _compute_next_position(self, component_returns, dates: pd.DatetimeIndex) -> float:
        if len(dates) == 0 or not component_returns:
            return 0.0

        signals = []
        for comp, ret in component_returns:
            signal = self._next_component_signal(ret, lag=comp["lag"], window=comp["window"])
            signals.append(signal)

        sig_values = np.asarray(signals, dtype=float)
        n_up = int(np.sum(sig_values > 0))
        n_down = int(np.sum(sig_values < 0))
        n_active = int(np.sum(sig_values != 0))
        if n_up <= n_down or n_active == 0:
            return 0.0

        vote_frac = n_up / n_active
        if vote_frac < CONVICTION_MIN:
            return 0.0
        return max(vote_frac**2, 0.0)

    def _next_component_signal(self, ret: pd.Series, *, lag: int, window: int) -> float:
        transformed = ret.rolling(window).sum() if window > 1 else ret
        if len(transformed) < lag:
            return 0.0
        value = transformed.iloc[-lag]
        if pd.isna(value):
            return 0.0
        return float(np.sign(value))

    def _normalize_component(self, component: str | dict) -> dict:
        if isinstance(component, str):
            return {
                "ticker": component,
                "field": "price",
                "lag": DEFAULT_LAG,
                "window": DEFAULT_WINDOW,
            }
        return {
            "ticker": component["ticker"],
            "field": component.get("field", "price"),
            "lag": int(component.get("lag", component.get("tau", DEFAULT_LAG))),
            "window": int(component.get("window", DEFAULT_WINDOW)),
        }
