"""Runtime validation for strategy engine outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_edge.engine.feed_contract import FeedContractError, validate_datetime_index


class SignalContractError(FeedContractError):
    """Raised when compute_signals() output violates the runtime contract."""


def _as_numeric_vector(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise SignalContractError(f"{name} must be a 1D numeric array.")
    if not np.isfinite(arr).all():
        raise SignalContractError(f"{name} contains non-finite numeric values.")
    return arr


def validate_signal_output(
    positions,
    dates,
    prices,
    *,
    profile: str = "daily",
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    pos_arr = _as_numeric_vector(positions, name="positions")
    price_arr = _as_numeric_vector(prices, name="prices")
    try:
        dates_idx = validate_datetime_index(dates, profile=profile, name="dates")
    except FeedContractError as exc:
        raise SignalContractError(str(exc)) from exc

    if len(pos_arr) != len(price_arr) or len(pos_arr) != len(dates_idx):
        raise SignalContractError(
            "positions, dates, and prices must have identical lengths."
        )
    return pos_arr, dates_idx, price_arr
