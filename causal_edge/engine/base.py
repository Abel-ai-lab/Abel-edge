"""Abstract base class for strategy engines.

Every strategy engine must implement compute_signals() and get_latest_signal().
Engines are standalone: strategies/ never imports causal_edge/ internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from causal_edge.engine.price_data import normalize_bars


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
        if self._bars_loader is None:
            raise RuntimeError("Price loader is not configured for this engine.")
        resolved_symbols = symbols or [
            self._price_data_config.get("symbol") or self.context.get("asset")
        ]
        df = self._bars_loader(
            symbols=resolved_symbols,
            start=start,
            end=end,
            timeframe=timeframe or self._price_data_config.get("timeframe", "1d"),
            limit=limit or self._price_data_config.get("limit"),
            fields=fields,
            config=self._price_data_config,
        )
        return normalize_bars(df)

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

