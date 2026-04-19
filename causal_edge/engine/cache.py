"""Market-data cache substrate for adapter-managed bar feeds."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

CACHE_ROOT_ENV = "CAUSAL_EDGE_CACHE_ROOT"
DEFAULT_CACHE_ROOT = Path(".cache/market_data")
_OPTION_EXCLUDE = {"env_path", "fields", "force", "cache_root"}


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


def cache_covers_request(
    metadata: dict[str, Any],
    *,
    start: object | None,
    end: object | None,
) -> bool:
    if not metadata:
        return False
    available_start = _as_timestamp((metadata.get("available_range") or {}).get("start"))
    available_end = _as_timestamp((metadata.get("available_range") or {}).get("end"))
    requested_start = _as_timestamp(start)
    requested_end = _as_timestamp(end)
    if requested_start is not None and (available_start is None or available_start > requested_start):
        return False
    if requested_end is not None and (available_end is None or available_end < requested_end):
        return False
    if requested_end is None and available_end is not None and not _is_recent_enough(available_end):
        return False
    return True


def write_cached_bars(entry: CacheEntry, bars: pd.DataFrame) -> dict[str, Any]:
    normalized = _normalize_cached_bars(bars)
    entry.data_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(entry.data_path, index=False)
    metadata = build_cache_metadata(entry, normalized)
    entry.meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def build_cache_metadata(entry: CacheEntry, bars: pd.DataFrame) -> dict[str, Any]:
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


def _as_timestamp(value: object | None) -> pd.Timestamp | None:
    if value in {None, ""}:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None


def _is_recent_enough(value: pd.Timestamp) -> bool:
    today = pd.Timestamp.now(tz=UTC).normalize()
    return value.normalize() >= today - pd.Timedelta(days=3)
