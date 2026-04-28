"""DecisionContext causal voting demo using Abel-discovered graph structure."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from abel_edge.engine.base import StrategyEngine

GRAPH_PATH = Path(__file__).parent / "causal_graph.json"
CONVICTION_MIN = 0.75  # ≥75% of votes must agree to trade
DEFAULT_LAG = 1
DEFAULT_WINDOW = 5


class CausalDemoEngine(StrategyEngine):
    """Vote-based strategy using named parent feeds plus target history."""

    def __init__(self, context: dict | None = None) -> None:
        super().__init__(context=context)
        with open(GRAPH_PATH, encoding="utf-8") as f:
            self.graph = json.load(f)

    def compute_decisions(self, ctx):
        """Generate vote-based next-position intent from graph components."""
        components = [
            self._normalize_component(item, "parent") for item in self.graph.get("parents", [])
        ]
        components += [
            self._normalize_component(item, "child") for item in self.graph.get("children", [])
        ]
        close = ctx.target.series("close").astype(float)
        target_ret = close.pct_change().fillna(0.0)

        sig_matrix = []
        available = set(ctx.available_feeds())
        for comp in components:
            ticker = comp["ticker"]
            tau = comp["lag"]
            win = comp["window"]
            if comp["type"] == "parent":
                if ticker not in available:
                    continue
                source = ctx.feed(ticker).asof_series("close").astype(float).pct_change().fillna(0.0)
            else:
                source = target_ret
            signal = source.rolling(win, min_periods=win).sum().shift(tau)
            sig_matrix.append(signal.fillna(0.0).apply(_sign_vote).to_numpy(dtype=float))

        if not sig_matrix:
            return ctx.decisions(pd.Series(0.0, index=close.index, dtype=float))

        sig_frame = pd.DataFrame(sig_matrix).to_numpy(dtype=float)
        n_up = (sig_frame > 0).sum(axis=0)
        n_down = (sig_frame < 0).sum(axis=0)
        n_active = (sig_frame != 0).sum(axis=0)
        vote_frac = pd.Series(
            n_up,
            index=close.index,
            dtype=float,
        )
        active_mask = n_active > 0
        vote_frac.loc[active_mask] = vote_frac.loc[active_mask] / n_active[active_mask]
        vote_frac.loc[~active_mask] = 0.5

        next_position = pd.Series(0.0, index=close.index, dtype=float)
        bull = n_up > n_down
        next_position.loc[bull] = vote_frac.loc[bull] ** 2
        weak = bull & (vote_frac < CONVICTION_MIN)
        next_position.loc[weak] = 0.0
        next_position = next_position.clip(lower=0.0, upper=1.0)
        return ctx.decisions(next_position)

    def _normalize_component(self, component: str | dict, default_role: str) -> dict:
        if isinstance(component, str):
            return {
                "ticker": component,
                "field": "price",
                "type": default_role,
                "lag": DEFAULT_LAG,
                "window": DEFAULT_WINDOW,
            }

        ticker = component["ticker"]
        field = component.get("field", "price")
        return {
            "ticker": ticker,
            "field": field,
            "type": component.get("type", default_role),
            "lag": int(component.get("lag", component.get("tau", DEFAULT_LAG))),
            "window": int(component.get("window", DEFAULT_WINDOW)),
        }


def resolve_price_column(df: pd.DataFrame, field: str) -> str:
    """Map Abel field names to local CSV column names."""
    if field == "price":
        for candidate in ("close", "price"):
            if candidate in df.columns:
                return candidate
        raise ValueError("Price data must contain either 'close' or 'price' column.")
    if field == "volume":
        if "volume" in df.columns:
            return "volume"
        raise ValueError("Volume data must contain 'volume' column.")
    raise ValueError(f"Unsupported Abel field '{field}'.")


def _sign_vote(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0
