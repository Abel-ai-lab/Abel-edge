"""Market-data cache substrate for adapter-managed bar feeds."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from abel_edge.engine.cache_lock import exclusive_cache_lock
from abel_edge.engine.point_in_time_series import is_date_only_bound

CACHE_ROOT_ENV = "ABEL_EDGE_CACHE_ROOT"
DEFAULT_CACHE_ROOT = Path(".cache/market_data")
_OPTION_EXCLUDE = {
    "env_path",
    "fields",
    "force",
    "cache_root",
    "max_cache_age_seconds",
}


@dataclass(frozen=True)
class CacheEntry:
    root: Path
    key: str
    adapter: str
    symbol: str
    timeframe: str
    data_path: Path
    meta_path: Path


def resolve_cache_root(explicit: str | Path | None = None) -> Path:
    value = explicit or os.environ.get(CACHE_ROOT_ENV) or DEFAULT_CACHE_ROOT
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def cache_entry_for_request(
    *,
    adapter: str,
    symbol: str,
    timeframe: str | None,
    profile: str,
    options: dict[str, Any] | None = None,
    cache_root: str | Path | None = None,
) -> CacheEntry:
    normalized_adapter = str(adapter or "").strip().lower()
    normalized_symbol = str(symbol or "").strip().upper()
    normalized_timeframe = str(timeframe or "1d").strip().lower()
    sanitized_options = _sanitize_options(options or {})
    payload = {
        "profile": str(profile or "daily").strip().lower(),
        "options": sanitized_options,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    root = resolve_cache_root(cache_root)
    entry_root = root / normalized_adapter / normalized_symbol / normalized_timeframe
    entry_root.mkdir(parents=True, exist_ok=True)
    return CacheEntry(
        root=root,
        key=digest,
        adapter=normalized_adapter,
        symbol=normalized_symbol,
        timeframe=normalized_timeframe,
        data_path=entry_root / f"{digest}.csv",
        meta_path=entry_root / f"{digest}.json",
    )


def point_in_time_cache_entry(
    *,
    adapter: str,
    series_spec_sha256: str,
    cache_root: str | Path | None = None,
) -> CacheEntry:
    """Resolve a path-safe cache entry keyed by the complete frozen series spec."""

    spec_hash = str(series_spec_sha256 or "").strip().lower()
    if len(spec_hash) != 64 or any(char not in "0123456789abcdef" for char in spec_hash):
        raise ValueError("point-in-time cache requires a lowercase series spec SHA-256")
    normalized_adapter = str(adapter or "").strip().lower()
    root = resolve_cache_root(cache_root)
    entry_root = root / normalized_adapter / "point_in_time_series"
    entry_root.mkdir(parents=True, exist_ok=True)
    disk_key = spec_hash[:40]
    return CacheEntry(
        root=root,
        key=spec_hash,
        adapter=normalized_adapter,
        symbol=spec_hash,
        timeframe="point_in_time_series",
        data_path=entry_root / f"{disk_key}.csv",
        meta_path=entry_root / f"{disk_key}.json",
    )


def load_cached_bars(entry: CacheEntry) -> pd.DataFrame | None:
    if not entry.data_path.exists():
        return None
    frame = pd.read_csv(entry.data_path)
    if frame.empty:
        return pd.DataFrame()
    return _normalize_cached_bars(frame)


def load_cached_metadata(entry: CacheEntry) -> dict[str, Any]:
    if not entry.meta_path.exists():
        return {}
    try:
        payload = json.loads(entry.meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def point_in_time_cache_covers_request(
    metadata: dict[str, Any],
    *,
    series_spec_sha256: str,
    start: object | None,
    end: object | None,
    limit: int | None,
) -> bool:
    if metadata.get("contract") != "abel-edge.point-in-time-cache/v2":
        return False
    if metadata.get("series_spec_sha256") != series_spec_sha256:
        return False
    requested = metadata.get("requested_range") or {}
    cached_start = _as_timestamp(requested.get("start"))
    cached_end = _as_timestamp(requested.get("end"))
    requested_start = _as_timestamp(start)
    requested_end = _as_timestamp(end)
    if bool(requested.get("end_is_inclusive_date")) != is_date_only_bound(end):
        return False
    if requested_start is not None and (
        cached_start is None or cached_start > requested_start
    ):
        return False
    if requested_end is not None and (cached_end is None or cached_end < requested_end):
        return False
    cached_limit = requested.get("limit")
    if cached_limit is not None and (
        cached_start != requested_start or cached_end != requested_end
    ):
        return False
    if limit is None:
        if cached_limit is not None:
            return False
    else:
        try:
            if cached_limit is not None and int(cached_limit) < int(limit):
                return False
        except (TypeError, ValueError):
            return False
    return bool(metadata.get("data_sha256"))


def load_cached_point_in_time_series(
    entry: CacheEntry,
    *,
    metadata: dict[str, Any],
) -> pd.DataFrame | None:
    if not entry.data_path.is_file():
        return None
    expected_data_hash = str(metadata.get("data_sha256") or "")
    if not expected_data_hash or _file_sha256(entry.data_path) != expected_data_hash:
        return None
    frame = pd.read_csv(entry.data_path)
    frame.attrs["source_receipt_sha256"] = str(metadata.get("source_receipt_sha256") or "")
    frame.attrs["series_spec_sha256"] = str(metadata.get("series_spec_sha256") or "")
    return frame


def cache_covers_request(
    metadata: dict[str, Any],
    *,
    start: object | None,
    end: object | None,
    limit: int | None = None,
    required_columns: Iterable[str] | None = None,
    max_cache_age_seconds: int | float | None = None,
) -> bool:
    if not metadata:
        return False
    if required_columns is not None:
        cached_columns = {str(column) for column in metadata.get("columns") or []}
        required = {str(column) for column in required_columns}
        if not required.issubset(cached_columns):
            return False
    if max_cache_age_seconds is not None:
        updated_at = _as_timestamp(metadata.get("updated_at"))
        if updated_at is None:
            return False
        max_age = pd.Timedelta(seconds=float(max_cache_age_seconds))
        if pd.Timestamp.now(tz=UTC) - updated_at > max_age:
            return False
    available_start = _as_timestamp((metadata.get("available_range") or {}).get("start"))
    available_end = _as_timestamp((metadata.get("available_range") or {}).get("end"))
    requested_start, requested_end = _as_timestamp(start), _as_timestamp(end)
    cached_request = metadata.get("requested_range") or {}
    cached_requested_start = _as_timestamp(cached_request.get("start"))
    cached_requested_end = _as_timestamp(cached_request.get("end"))
    if requested_start is not None:
        if available_start is None:
            return False
        if available_start > requested_start:
            if cached_requested_start is None or cached_requested_start > requested_start:
                return False
    if requested_end is not None:
        end_was_probed = cached_requested_end is not None and cached_requested_end >= requested_end
        if available_end is None or (available_end < requested_end and not end_was_probed):
            return False
    if limit is not None:
        try:
            requested_limit = int(limit)
            row_count = int(metadata.get("row_count") or 0)
            cached_requested_limit = int((cached_request or {}).get("limit") or 0)
        except (TypeError, ValueError):
            return False
        if (
            requested_limit > 0
            and row_count < requested_limit
            and cached_requested_limit < requested_limit
        ):
            return False
    return True


def write_cached_bars(
    entry: CacheEntry,
    bars: pd.DataFrame,
    *,
    requested_start: object | None = None,
    requested_end: object | None = None,
    requested_limit: int | None = None,
) -> dict[str, Any]:
    normalized = _normalize_cached_bars(bars)
    entry.data_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(entry.data_path, index=False)
    metadata = build_cache_metadata(
        entry,
        normalized,
        requested_start=requested_start,
        requested_end=requested_end,
        requested_limit=requested_limit,
    )
    entry.meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def write_cached_point_in_time_series(
    entry: CacheEntry,
    frame: pd.DataFrame,
    *,
    series_spec_sha256: str,
    source_receipt_sha256: str,
    requested_start: object | None,
    requested_end: object | None,
    requested_limit: int | None,
) -> dict[str, Any]:
    """Persist a point-in-time response with its materialization identities."""

    if "value" not in frame.columns or not {
        "event_time",
        "timestamp",
    }.intersection(frame.columns):
        raise ValueError(
            "Cached point-in-time series requires value and event_time or timestamp."
        )
    entry.data_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_cache_lock(entry.data_path):
        temporary = entry.data_path.parent / f".data-{os.urandom(8).hex()}.partial"
        frame.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(entry.data_path)
        metadata = {
            "contract": "abel-edge.point-in-time-cache/v2",
            "adapter": entry.adapter,
            "cache_key": entry.key,
            "data_path": str(entry.data_path),
            "metadata_path": str(entry.meta_path),
            "data_sha256": _file_sha256(entry.data_path),
            "series_spec_sha256": series_spec_sha256,
            "source_receipt_sha256": source_receipt_sha256,
            "requested_range": {
                "start": _format_request_bound(requested_start, exact=True),
                "end": _format_request_bound(requested_end, exact=True),
                "end_is_inclusive_date": is_date_only_bound(requested_end),
                "limit": int(requested_limit) if requested_limit is not None else None,
            },
            "row_count": int(len(frame)),
            "columns": list(frame.columns),
            "updated_at": datetime.now(tz=UTC).isoformat(),
        }
        metadata_temporary = entry.meta_path.parent / f".meta-{os.urandom(8).hex()}.partial"
        metadata_temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metadata_temporary.replace(entry.meta_path)
    return metadata


def build_cache_metadata(
    entry: CacheEntry,
    bars: pd.DataFrame,
    *,
    requested_start: object | None = None,
    requested_end: object | None = None,
    requested_limit: int | None = None,
) -> dict[str, Any]:
    normalized = _normalize_cached_bars(bars)
    start = None
    end = None
    if not normalized.empty:
        timestamps = pd.to_datetime(normalized["timestamp"], utc=True)
        start = timestamps.min().date().isoformat()
        end = timestamps.max().date().isoformat()
    return {
        "adapter": entry.adapter,
        "symbol": entry.symbol,
        "timeframe": entry.timeframe,
        "cache_key": entry.key,
        "data_path": str(entry.data_path),
        "metadata_path": str(entry.meta_path),
        "available_range": {
            "start": start,
            "end": end,
        },
        "requested_range": {
            "start": _format_request_bound(requested_start),
            "end": _format_request_bound(requested_end),
            "limit": int(requested_limit) if requested_limit is not None else None,
        },
        "row_count": int(len(normalized)),
        "columns": list(normalized.columns),
        "updated_at": datetime.now(tz=UTC).isoformat(),
    }


def _normalize_cached_bars(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "timestamp" not in normalized.columns:
        raise ValueError("Cached bars must include a 'timestamp' column.")
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"]).sort_values("timestamp")
    if "symbol" in normalized.columns:
        normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
    normalized = normalized.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    normalized["timestamp"] = normalized["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return normalized.reset_index(drop=True)


def _sanitize_options(options: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in sorted(options.items()):
        if key in _OPTION_EXCLUDE or value is None:
            continue
        payload[str(key)] = value
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_request_bound(value: object | None, *, exact: bool = False) -> str | None:
    timestamp = _as_timestamp(value)
    if timestamp is None:
        return None
    return timestamp.isoformat() if exact else timestamp.date().isoformat()


def _as_timestamp(value: object | None) -> pd.Timestamp | None:
    if value in {None, ""}:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
