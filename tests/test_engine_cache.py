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
