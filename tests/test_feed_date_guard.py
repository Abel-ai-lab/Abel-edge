"""Tests for hidden max-data-date guard primitives."""

from __future__ import annotations

import pandas as pd
import pytest

from abel_edge.engine.feed_contract import (
    DATE_GUARD_MODE_ENV,
    MAX_DATA_DATE_ENV,
    FeedDateGuardError,
    apply_max_data_date_guard,
    assert_frame_respects_max_data_date,
    guarded_max_data_date,
)
from abel_edge.engine.feed_loader import load_feed_frame
from abel_edge.engine.adapter_registry import AbelDataFeedAdapter, CSVDataFeedAdapter, FeedLoadRequest
from abel_edge.engine.cache import cache_entry_for_request, write_cached_bars


def test_guarded_max_data_date_is_disabled_without_cutoff():
    assert guarded_max_data_date({}) is None


def test_apply_max_data_date_guard_closes_open_ended_request():
    env = {
        MAX_DATA_DATE_ENV: "2026-06-15",
        DATE_GUARD_MODE_ENV: "fail-closed",
    }

    assert apply_max_data_date_guard(None, source="unit-test", environ=env) == "2026-06-15"


def test_apply_max_data_date_guard_rejects_request_after_cutoff():
    env = {
        MAX_DATA_DATE_ENV: "2026-06-15",
        DATE_GUARD_MODE_ENV: "fail-closed",
    }

    with pytest.raises(FeedDateGuardError, match="date_guard_violation"):
        apply_max_data_date_guard("2026-06-16", source="unit-test", environ=env)


def test_apply_max_data_date_guard_allows_request_on_cutoff():
    env = {
        MAX_DATA_DATE_ENV: "2026-06-15",
        DATE_GUARD_MODE_ENV: "fail-closed",
    }

    assert (
        apply_max_data_date_guard("2026-06-15T18:00:00Z", source="unit-test", environ=env)
        == "2026-06-15T18:00:00Z"
    )


def test_assert_frame_respects_max_data_date_detects_polluted_cache():
    env = {
        MAX_DATA_DATE_ENV: "2026-06-15",
        DATE_GUARD_MODE_ENV: "fail-closed",
    }
    frame = pd.DataFrame({"timestamp": ["2026-06-15T00:00:00Z", "2026-06-16T00:00:00Z"]})

    with pytest.raises(FeedDateGuardError, match="polluted_cache"):
        assert_frame_respects_max_data_date(frame, source="cached bars", environ=env)


def test_assert_frame_respects_max_data_date_can_be_disabled():
    env = {
        MAX_DATA_DATE_ENV: "2026-06-15",
        DATE_GUARD_MODE_ENV: "off",
    }
    frame = pd.DataFrame({"timestamp": ["2026-06-16T00:00:00Z"]})

    assert_frame_respects_max_data_date(frame, source="cached bars", environ=env)


def test_load_feed_frame_detects_polluted_csv_cache_before_filtering(tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_DATA_DATE_ENV, "2026-06-15")
    monkeypatch.setenv(DATE_GUARD_MODE_ENV, "fail-closed")
    csv_path = tmp_path / "aapl.csv"
    csv_path.write_text(
        "timestamp,close\n"
        "2026-06-15T00:00:00Z,100\n"
        "2026-06-16T00:00:00Z,101\n",
        encoding="utf-8",
    )

    with pytest.raises(FeedDateGuardError, match="polluted_cache"):
        load_feed_frame(
            {
                "name": "primary",
                "kind": "bars",
                "adapter": "csv",
                "path": str(csv_path),
                "symbol": "AAPL",
            },
            end="2026-06-15",
        )


def test_load_feed_frame_closes_open_ended_abel_request(tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_DATA_DATE_ENV, "2026-06-15")
    monkeypatch.setenv(DATE_GUARD_MODE_ENV, "fail-closed")
    calls = []

    def fake_fetch_bars(
        *,
        symbols,
        start=None,
        end=None,
        timeframe="1d",
        limit=None,
        fields=None,
        config=None,
    ):
        calls.append({"symbols": symbols, "start": start, "end": end, "limit": limit})
        return pd.DataFrame(
            {
                "timestamp": ["2026-06-15T00:00:00Z"],
                "symbol": ["AAPL"],
                "open": [99.0],
                "high": [101.0],
                "low": [98.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )

    import abel_edge.plugins.abel.prices as prices_module

    monkeypatch.setattr(prices_module, "fetch_bars", fake_fetch_bars)
    load_feed_frame(
        {
            "name": "primary",
            "kind": "bars",
            "adapter": "abel",
            "symbol": "AAPL",
            "cache_root": str(tmp_path),
        }
    )

    assert calls[0]["end"] == "2026-06-15"


def test_fetch_bars_wrapper_applies_guarded_end(monkeypatch):
    monkeypatch.setenv("ABEL_API_KEY", "abel_test")
    monkeypatch.setenv(MAX_DATA_DATE_ENV, "2026-06-15")
    monkeypatch.setenv(DATE_GUARD_MODE_ENV, "fail-closed")
    observed = {}

    def fake_fetch_bars(self, **kwargs):
        observed["end"] = kwargs["end"]
        return [{"timestamp": "2026-06-15T00:00:00Z", "symbol": "AAPL", "close": 100.0}]

    from abel_edge.plugins.abel.client import AbelClient
    from abel_edge.plugins.abel import prices as prices_module

    monkeypatch.setattr(AbelClient, "fetch_bars", fake_fetch_bars)

    prices_module.fetch_bars(symbols=["AAPL"])

    assert observed["end"] == "2026-06-15"


def test_abel_adapter_detects_polluted_cached_bars(tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_DATA_DATE_ENV, "2026-06-15")
    monkeypatch.setenv(DATE_GUARD_MODE_ENV, "fail-closed")
    options = {"cache_root": str(tmp_path)}
    request = FeedLoadRequest(
        adapter="abel",
        kind="bars",
        symbol="AAPL",
        field=None,
        timeframe="1d",
        start=None,
        end=None,
        limit=None,
        profile="daily",
        options=options,
        strategy_id=None,
        feed_name="warm-cache:AAPL",
    )
    entry = cache_entry_for_request(
        adapter="abel",
        symbol="AAPL",
        timeframe="1d",
        profile="daily",
        options=options,
        cache_root=tmp_path,
    )
    write_cached_bars(
        entry,
        pd.DataFrame(
            {
                "timestamp": ["2026-06-15T00:00:00Z", "2026-06-16T00:00:00Z"],
                "symbol": ["AAPL", "AAPL"],
                "open": [99.0, 100.0],
                "high": [101.0, 102.0],
                "low": [98.0, 99.0],
                "close": [100.0, 101.0],
                "volume": [1000.0, 1100.0],
            }
        ),
        requested_start=None,
        requested_end=None,
    )

    with pytest.raises(FeedDateGuardError, match="polluted_cache"):
        AbelDataFeedAdapter().load(request)


def test_csv_adapter_detects_polluted_direct_load(tmp_path, monkeypatch):
    monkeypatch.setenv(MAX_DATA_DATE_ENV, "2026-06-15")
    monkeypatch.setenv(DATE_GUARD_MODE_ENV, "fail-closed")
    csv_path = tmp_path / "aapl.csv"
    csv_path.write_text(
        "timestamp,symbol,close\n"
        "2026-06-15T00:00:00Z,AAPL,100\n"
        "2026-06-16T00:00:00Z,AAPL,101\n",
        encoding="utf-8",
    )
    request = FeedLoadRequest(
        adapter="csv",
        kind="bars",
        symbol="AAPL",
        field=None,
        timeframe="1d",
        start=None,
        end=None,
        limit=None,
        profile="daily",
        options={"path": str(csv_path)},
        strategy_id=None,
        feed_name="warm-cache:AAPL",
    )

    with pytest.raises(FeedDateGuardError, match="polluted_cache"):
        CSVDataFeedAdapter().load(request)
