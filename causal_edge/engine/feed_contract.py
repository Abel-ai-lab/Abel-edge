"""Feed and datetime contract helpers for engine-side runtime validation."""

from __future__ import annotations

import pandas as pd


class FeedContractError(ValueError):
    """Base exception for feed contract violations."""


class FeedNormalizationError(FeedContractError):
    """Raised when a feed cannot be normalized into the runtime contract."""


class FeedAlignmentError(FeedContractError):
    """Raised when a series cannot be aligned safely to strategy dates."""


SUPPORTED_DATA_PROFILES = {"daily"}


def validate_data_profile(profile: str) -> str:
    value = str(profile or "").strip().lower()
    if value not in SUPPORTED_DATA_PROFILES:
        raise FeedContractError(
            f"Unsupported data contract profile '{profile}'. "
            f"Supported: {sorted(SUPPORTED_DATA_PROFILES)}."
        )
    return value


def validate_datetime_index(
    dates,
    *,
    profile: str = "daily",
    name: str = "dates",
) -> pd.DatetimeIndex:
    resolved_profile = validate_data_profile(profile)
    idx = pd.DatetimeIndex(dates)
    if idx.tz is None:
        raise FeedNormalizationError(
            f"{name} must be UTC-aware for the supported {resolved_profile!r} contract."
        )
    idx = idx.tz_convert("UTC")
    if idx.hasnans:
        raise FeedNormalizationError(f"{name} contains NaT values.")
    if resolved_profile == "daily" and not idx.equals(idx.normalize()):
        raise FeedNormalizationError(
            f"{name} must be normalized to midnight UTC for the supported 'daily' contract."
        )
    if not idx.is_monotonic_increasing:
        raise FeedNormalizationError(f"{name} must be sorted in strictly increasing time order.")
    if idx.has_duplicates:
        raise FeedNormalizationError(f"{name} contains duplicate timestamps.")
    return idx


def normalize_series_frame(
    df: pd.DataFrame,
    *,
    field: str,
    name: str,
    profile: str = "daily",
) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise FeedNormalizationError(f"{name} is missing required column 'timestamp'.")
    if field not in df.columns:
        raise FeedNormalizationError(f"{name} is missing required column '{field}'.")

    frame = df.copy()
    frame["timestamp"] = validate_datetime_index(
        frame["timestamp"],
        profile=profile,
        name=f"{name}.timestamp",
    )
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str)
    frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if frame[field].isna().any():
        raise FeedNormalizationError(f"{name}.{field} contains non-numeric values.")

    sort_cols = ["timestamp"] if "symbol" not in frame.columns else ["symbol", "timestamp"]
    frame = frame.sort_values(sort_cols).reset_index(drop=True)
    dup_cols = ["timestamp"] if "symbol" not in frame.columns else ["symbol", "timestamp"]
    if frame.duplicated(subset=dup_cols).any():
        raise FeedNormalizationError(f"{name} contains duplicate timestamps in the runtime contract.")
    return frame


def align_series_to_dates(
    series: pd.Series,
    dates,
    *,
    profile: str = "daily",
    method: str | None = "ffill",
    allow_gaps: bool = True,
    name: str = "series",
) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if not isinstance(series.index, pd.DatetimeIndex):
        raise FeedAlignmentError(f"{name} must use a DatetimeIndex before alignment.")
    try:
        source_idx = validate_datetime_index(series.index, profile=profile, name=f"{name}.index")
        target_idx = validate_datetime_index(dates, profile=profile, name="strategy dates")
    except FeedNormalizationError as exc:
        raise FeedAlignmentError(str(exc)) from exc
    normalized = series.copy()
    normalized.index = source_idx
    normalized = normalized.sort_index()

    if method not in {None, "ffill"}:
        raise FeedAlignmentError(
            f"Unsupported alignment method '{method}'. Supported: None, 'ffill'."
        )

    aligned = normalized.reindex(target_idx)
    if method == "ffill":
        aligned = aligned.ffill()
    if not allow_gaps and aligned.isna().any():
        raise FeedAlignmentError(
            f"{name} could not be aligned to strategy dates without gaps."
        )
    return aligned
