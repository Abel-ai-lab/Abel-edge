"""Feed and datetime contract helpers for engine-side runtime validation."""

from __future__ import annotations

import os
from collections.abc import Mapping

import pandas as pd


MAX_DATA_DATE_ENV = "ABEL_EDGE_MAX_DATA_DATE"
DATE_GUARD_MODE_ENV = "ABEL_EDGE_DATE_GUARD_MODE"


class FeedContractError(ValueError):
    """Base exception for feed contract violations."""


class FeedNormalizationError(FeedContractError):
    """Raised when a feed cannot be normalized into the runtime contract."""


class FeedAlignmentError(FeedContractError):
    """Raised when a series cannot be aligned safely to strategy dates."""


class FeedDateGuardError(FeedContractError):
    """Raised when guarded data access exceeds the configured max data date."""


SUPPORTED_DATA_PROFILES = {"daily"}


def guarded_max_data_date(environ: Mapping[str, str] | None = None) -> pd.Timestamp | None:
    env = environ if environ is not None else os.environ
    mode = str(env.get(DATE_GUARD_MODE_ENV) or "fail-closed").strip().lower()
    if mode in {"", "off", "disabled", "disable", "false", "0"}:
        return None
    if mode != "fail-closed":
        raise FeedDateGuardError(
            f"Unsupported {DATE_GUARD_MODE_ENV}={mode!r}; supported modes are "
            "'fail-closed' and 'off'."
        )

    raw = str(env.get(MAX_DATA_DATE_ENV) or "").strip()
    if not raw:
        return None
    try:
        cutoff = pd.to_datetime(raw, utc=True)
    except (TypeError, ValueError) as exc:
        raise FeedDateGuardError(f"{MAX_DATA_DATE_ENV} must be a valid date.") from exc
    if pd.isna(cutoff):
        raise FeedDateGuardError(f"{MAX_DATA_DATE_ENV} must be a valid date.")
    return pd.Timestamp(cutoff).normalize()


def apply_max_data_date_guard(
    end,
    *,
    source: str,
    environ: Mapping[str, str] | None = None,
):
    cutoff = guarded_max_data_date(environ)
    if cutoff is None:
        return end
    if end is None:
        return cutoff.date().isoformat()

    requested_end = _guard_timestamp(end, name="requested end")
    if requested_end > cutoff:
        raise FeedDateGuardError(
            f"date_guard_violation: {source} requested end {requested_end.date().isoformat()} "
            f"after {MAX_DATA_DATE_ENV}={cutoff.date().isoformat()}."
        )
    return end


def assert_frame_respects_max_data_date(
    frame: pd.DataFrame,
    *,
    source: str,
    timestamp_col: str = "timestamp",
    environ: Mapping[str, str] | None = None,
) -> None:
    cutoff = guarded_max_data_date(environ)
    if cutoff is None or frame.empty or timestamp_col not in frame.columns:
        return
    observed = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    if observed.isna().all():
        return
    max_observed = pd.Timestamp(observed.max()).normalize()
    if max_observed > cutoff:
        raise FeedDateGuardError(
            f"polluted_cache: {source} observed data through "
            f"{max_observed.date().isoformat()}, after {MAX_DATA_DATE_ENV}="
            f"{cutoff.date().isoformat()}. Clear the affected cache or run in a clean workspace."
        )


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
    assume_utc_for_naive: bool = False,
) -> pd.DatetimeIndex:
    resolved_profile = validate_data_profile(profile)
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(dates, utc=False))
    except (TypeError, ValueError) as exc:
        raise FeedNormalizationError(f"{name} contains invalid timestamp values.") from exc
    if idx.tz is None:
        if assume_utc_for_naive:
            idx = idx.tz_localize("UTC")
        else:
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
    assume_utc_for_naive: bool = False,
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
        assume_utc_for_naive=assume_utc_for_naive,
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


def _guard_timestamp(value, *, name: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError) as exc:
        raise FeedDateGuardError(f"{name} must be a valid date.") from exc
    if pd.isna(ts):
        raise FeedDateGuardError(f"{name} must be a valid date.")
    return pd.Timestamp(ts).normalize()
