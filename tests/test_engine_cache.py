from datetime import UTC, datetime, timedelta

import pandas as pd

from abel_edge.engine.cache import (
    cache_covers_request,
    cache_entry_for_request,
    write_cached_bars,
)


def test_write_cached_bars_records_requested_range(tmp_path):
    entry = cache_entry_for_request(
        adapter="abel",
        symbol="AAPL",
        timeframe="1d",
        profile="daily",
        cache_root=tmp_path,
    )
    bars = pd.DataFrame(
        {
            "timestamp": ["2020-01-02T00:00:00Z"],
            "symbol": ["AAPL"],
            "close": [100.0],
        }
    )

    metadata = write_cached_bars(
        entry,
        bars,
        requested_start="2020-01-01",
        requested_end=None,
    )

    assert metadata["requested_range"] == {"start": "2020-01-01", "end": None}


def test_cache_covers_partial_history_when_request_boundary_was_probed():
    metadata = {
        "available_range": {"start": "2020-01-02", "end": "2099-01-01"},
        "requested_range": {"start": "2020-01-01", "end": None},
    }

    assert cache_covers_request(metadata, start="2020-01-01", end=None)
    assert not cache_covers_request(metadata, start="2019-12-31", end=None)


def test_cache_rejects_partial_history_without_available_start():
    metadata = {
        "available_range": {"start": None, "end": "2099-01-01"},
        "requested_range": {"start": "2020-01-01", "end": None},
    }

    assert not cache_covers_request(metadata, start="2020-01-01", end=None)


def test_cache_with_open_end_does_not_require_recent_available_end():
    metadata = {
        "available_range": {"start": "2020-01-01", "end": "2024-01-01"},
        "requested_range": {"start": "2020-01-01", "end": None},
    }

    assert cache_covers_request(metadata, start="2020-01-01", end=None)


def test_cache_rejects_missing_required_columns():
    metadata = {
        "available_range": {"start": "2020-01-01", "end": "2024-01-01"},
        "requested_range": {"start": "2020-01-01", "end": None},
        "columns": ["timestamp", "symbol", "close"],
    }

    assert not cache_covers_request(
        metadata,
        start="2020-01-01",
        end=None,
        required_columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    )


def test_cache_accepts_required_columns_when_present():
    metadata = {
        "available_range": {"start": "2020-01-01", "end": "2024-01-01"},
        "requested_range": {"start": "2020-01-01", "end": None},
        "columns": ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    }

    assert cache_covers_request(
        metadata,
        start="2020-01-01",
        end=None,
        required_columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    )


def test_cache_max_age_uses_metadata_updated_at():
    fresh_metadata = {
        "available_range": {"start": "2020-01-01", "end": "2024-01-01"},
        "requested_range": {"start": "2020-01-01", "end": None},
        "updated_at": datetime.now(tz=UTC).isoformat(),
    }
    stale_metadata = {
        **fresh_metadata,
        "updated_at": (datetime.now(tz=UTC) - timedelta(days=2)).isoformat(),
    }

    assert cache_covers_request(
        fresh_metadata,
        start="2020-01-01",
        end=None,
        max_cache_age_seconds=86400,
    )
    assert not cache_covers_request(
        stale_metadata,
        start="2020-01-01",
        end=None,
        max_cache_age_seconds=86400,
    )
