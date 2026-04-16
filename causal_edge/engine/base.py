"""Abstract base class for strategy engines.

Every strategy engine must implement compute_signals() and get_latest_signal().
Engines are standalone: strategies/ never imports causal_edge/ internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from causal_edge.engine.feed_contract import align_series_to_dates
from causal_edge.engine.feed_loader import load_declared_feed
from causal_edge.engine.signal_contract import validate_signal_output


class StrategyEngine(ABC):
    """Base class for all strategy engines.

    Subclasses must implement:
        compute_signals() -> tuple of (positions, dates, prices)
        get_latest_signal() -> dict with at least 'position' key
    """

    def __init__(self, context: dict | None = None) -> None:
        self.context = context
        self._bars_loader = None
        self._price_data_config = {}

    def bind_price_loader(self, loader, price_data_config: dict | None = None) -> None:
        self._bars_loader = loader
        self._price_data_config = price_data_config or {}

    def load_bars(
        self,
        symbols: list[str] | None = None,
        *,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        bars = self.load_feed(
            "primary",
            start=start,
            end=end,
            timeframe=timeframe,
            limit=limit,
            fields=fields,
        )
        if symbols is None:
            return bars
        return bars[bars["symbol"].isin(symbols)].reset_index(drop=True)

    def load_feed(
        self,
        name: str,
        *,
        start=None,
        end=None,
        timeframe: str | None = None,
        limit: int | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        return load_declared_feed(
            self,
            name,
            start=start,
            end=end,
            timeframe=timeframe,
            limit=limit,
            fields=fields,
        )

    def feed_series(
        self,
        name: str,
        field: str = "close",
        *,
        align_to=None,
        method: str | None = None,
        allow_gaps: bool = True,
    ) -> pd.Series:
        frame = self.load_feed(name)
        feed_cfg = ((self.context or {}).get("_feeds") or {}).get(name) or {}
        kind = feed_cfg.get("kind")
        if kind == "series":
            series = pd.Series(frame["value"].to_numpy(), index=pd.DatetimeIndex(frame["timestamp"]))
        else:
            if field not in frame.columns:
                raise ValueError(f"Feed '{name}' does not expose field '{field}'.")
            series = pd.Series(frame[field].to_numpy(), index=pd.DatetimeIndex(frame["timestamp"]))
        if align_to is not None:
            series = self.align_series(
                series,
                align_to,
                method="ffill" if method is None else method,
                allow_gaps=allow_gaps,
            )
        return series

    def align_series(
        self,
        series: pd.Series,
        dates,
        *,
        method: str | None = "ffill",
        allow_gaps: bool = True,
    ) -> pd.Series:
        profile = ((self.context or {}).get("_data_contract") or {}).get("profile", "daily")
        return align_series_to_dates(
            series,
            dates,
            profile=profile,
            method=method,
            allow_gaps=allow_gaps,
        )

    def finalize_signals(
        self,
        positions,
        dates,
        prices,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        profile = ((self.context or {}).get("_data_contract") or {}).get("profile", "daily")
        return validate_signal_output(
            positions,
            dates,
            prices,
            profile=profile,
        )

    @abstractmethod
    def compute_signals(
        self,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        """Compute full signal history.

        Returns:
            Tuple of (positions, dates, prices) where:
                positions: np.ndarray of daily position sizes (0=flat, 1=long).
                    IMPORTANT: positions[t] must be decided using only data through
                    day t-1. Apply .shift(1) to any indicators before using them
                    to determine positions. This prevents look-ahead bias.
                dates: pd.DatetimeIndex of trading dates
                prices: np.ndarray of daily closing prices
        """

    @abstractmethod
    def get_latest_signal(self) -> dict:
        """Return the most recent signal as a dict.

        Must include at least a 'position' key.
        """

    def get_paper_signal(self, *, as_of=None) -> dict:
        """Return the next position decided at the close of ``as_of``.

        Engines can override this to support incrementally appending live paper-trading rows.
        The returned dict must include at least ``next_position``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement paper trading. "
            "Add get_paper_signal(as_of=...) to the engine."
        )
