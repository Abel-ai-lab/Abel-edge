"""Range-aware disk cache for Abel point-in-time series."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

import pandas as pd

from abel_edge.engine.adapter_registry import (
    AbelDataFeedAdapter,
    FeedLoadRequest,
    _point_in_time_cache_end_has_elapsed,
)
from abel_edge.engine.cache import (
    point_in_time_cache_covers_request,
    point_in_time_cache_entry,
    write_cached_point_in_time_series,
)
from abel_edge.plugins.abel import compile_cap_node_series_spec

SOURCE_IDENTITY = "https://cap.example/api"


def _point_spec():
    return compile_cap_node_series_spec(
        node_id="health.openfda.drug.events:event_count#96bc3e82",
        graph_ref={"graph_id": "abel-main", "graph_version": "CausalNodeV4"},
        source_receipt_sha256="e" * 64,
    )


def test_builtin_abel_adapter_reuses_exact_point_in_time_cache(tmp_path, monkeypatch):
    point_spec = _point_spec()
    calls = []

    class CanonicalModule:
        @staticmethod
        def load_cap_node_series(**kwargs):
            calls.append(kwargs)
            frame = pd.DataFrame(
                {
                    "event_time": ["2024-01-02T05:00:00Z"],
                    "timestamp": ["2024-01-02T05:00:00Z"],
                    "value": [0.25],
                }
            )
            frame.attrs["source_receipt_sha256"] = "e" * 64
            frame.attrs["series_spec_sha256"] = point_spec.sha256
            return frame

    real_import = importlib.import_module

    def fake_import(name):
        if name == "abel_edge.plugins.abel.cap_node_series":
            return CanonicalModule()
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    request = FeedLoadRequest(
        adapter="abel",
        kind="point_in_time_series",
        symbol=None,
        field=None,
        timeframe=None,
        start="2024-01-01",
        end="2024-01-31",
        limit=None,
        profile="daily",
        options={"cache_root": str(tmp_path)},
        strategy_id="demo",
        feed_name="graph_parent_01",
        series_spec=point_spec,
    )

    first = AbelDataFeedAdapter().load(request)
    second = AbelDataFeedAdapter().load(request)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert second.attrs["source_receipt_sha256"] == "e" * 64
    assert second.attrs["series_spec_sha256"] == point_spec.sha256


def test_builtin_abel_adapter_rejects_other_cap_endpoint_cache(tmp_path, monkeypatch):
    point_spec = _point_spec()
    calls = []

    class CanonicalModule:
        @staticmethod
        def load_cap_node_series(**kwargs):
            calls.append(kwargs)
            frame = pd.DataFrame(
                {
                    "event_time": ["2024-01-02T05:00:00Z"],
                    "timestamp": ["2024-01-02T05:00:00Z"],
                    "value": [0.25],
                }
            )
            frame.attrs["source_receipt_sha256"] = "e" * 64
            frame.attrs["series_spec_sha256"] = point_spec.sha256
            return frame

    real_import = importlib.import_module

    def fake_import(name):
        if name == "abel_edge.plugins.abel.cap_node_series":
            return CanonicalModule()
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    request = FeedLoadRequest(
        adapter="abel",
        kind="point_in_time_series",
        symbol=None,
        field=None,
        timeframe=None,
        start=None,
        end="2024-01-31",
        limit=None,
        profile="daily",
        options={"cache_root": str(tmp_path)},
        strategy_id="demo",
        feed_name="graph_parent_01",
        series_spec=point_spec,
    )
    adapter = AbelDataFeedAdapter()
    monkeypatch.setenv("ABEL_CAP_BASE_URL", "https://cap-one.example/api")
    adapter.load(request)
    adapter.load(request)
    monkeypatch.setenv("ABEL_CAP_BASE_URL", "https://cap-two.example/api")
    adapter.load(request)
    adapter.load(request)

    assert len(calls) == 2


def test_point_in_time_cache_end_must_have_elapsed():
    now = pd.Timestamp("2026-05-01T12:00:00Z")

    assert not _point_in_time_cache_end_has_elapsed("2026-05-01", now=now)
    assert _point_in_time_cache_end_has_elapsed("2026-04-30", now=now)
    assert not _point_in_time_cache_end_has_elapsed("2026-05-01T13:00:00Z", now=now)
    assert _point_in_time_cache_end_has_elapsed("2026-05-01T11:00:00Z", now=now)


def test_limited_point_in_time_cache_does_not_cover_unlimited_request():
    metadata = {
        "contract": "abel-edge.point-in-time-cache/v1",
        "series_spec_sha256": "a" * 64,
        "source_receipt_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "requested_range": {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "limit": 30,
        },
    }

    assert not point_in_time_cache_covers_request(
        metadata,
        series_spec_sha256="a" * 64,
        source_identity=SOURCE_IDENTITY,
        start="2024-01-01",
        end="2024-12-31",
        limit=None,
    )


def test_limited_point_in_time_cache_does_not_cover_different_bounds():
    metadata = {
        "contract": "abel-edge.point-in-time-cache/v1",
        "series_spec_sha256": "a" * 64,
        "source_receipt_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "requested_range": {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "limit": 30,
        },
    }

    assert not point_in_time_cache_covers_request(
        metadata,
        series_spec_sha256="a" * 64,
        source_identity=SOURCE_IDENTITY,
        start="2024-01-01",
        end="2024-06-30",
        limit=10,
    )


def test_point_in_time_cache_preserves_intraday_request_bounds(tmp_path):
    spec_hash = "a" * 64
    receipt_hash = "b" * 64
    entry = point_in_time_cache_entry(
        adapter="abel",
        series_spec_sha256=spec_hash,
        cache_root=tmp_path,
    )
    metadata = write_cached_point_in_time_series(
        entry,
        pd.DataFrame(
            {
                "event_time": ["2026-05-01T12:00:00Z"],
                "value": [1.0],
            }
        ),
        series_spec_sha256=spec_hash,
        source_identity=SOURCE_IDENTITY,
        source_receipt_sha256=receipt_hash,
        requested_start="2026-05-01T12:00:00Z",
        requested_end="2026-05-01T18:00:00Z",
        requested_limit=None,
    )

    assert metadata["requested_range"] == {
        "start": "2026-05-01T12:00:00+00:00",
        "end": "2026-05-01T18:00:00+00:00",
        "end_is_inclusive_date": False,
        "limit": None,
    }
    assert metadata["source_identity"] == SOURCE_IDENTITY
    assert not point_in_time_cache_covers_request(
        metadata,
        series_spec_sha256=spec_hash,
        source_identity=SOURCE_IDENTITY,
        start="2026-05-01T06:00:00Z",
        end="2026-05-01T18:00:00Z",
        limit=None,
    )
    assert not point_in_time_cache_covers_request(
        metadata,
        series_spec_sha256=spec_hash,
        source_identity=SOURCE_IDENTITY,
        start="2026-05-01T12:00:00Z",
        end="2026-05-01T20:00:00Z",
        limit=None,
    )
    assert not point_in_time_cache_covers_request(
        metadata,
        series_spec_sha256=spec_hash,
        source_identity=SOURCE_IDENTITY,
        start="2026-05-01T12:00:00Z",
        end="2026-05-01",
        limit=None,
    )


def test_point_in_time_cache_coverage_does_not_require_artifact_receipt(tmp_path):
    spec_hash = "a" * 64
    entry = point_in_time_cache_entry(
        adapter="abel",
        series_spec_sha256=spec_hash,
        cache_root=tmp_path,
    )
    metadata = write_cached_point_in_time_series(
        entry,
        pd.DataFrame(
            {
                "event_time": ["2026-05-01T12:00:00Z"],
                "value": [1.0],
            }
        ),
        series_spec_sha256=spec_hash,
        source_identity=SOURCE_IDENTITY,
        source_receipt_sha256="b" * 64,
        requested_start="2026-05-01",
        requested_end="2026-05-31",
        requested_limit=None,
    )

    assert metadata["requested_range"]["end_is_inclusive_date"] is True
    assert point_in_time_cache_covers_request(
        metadata,
        series_spec_sha256=spec_hash,
        source_identity=SOURCE_IDENTITY,
        start="2026-05-01",
        end="2026-05-31",
        limit=None,
    )


def test_point_in_time_cache_rejects_legacy_day_granularity_metadata():
    metadata = {
        "contract": "abel-edge.point-in-time-cache/v1",
        "series_spec_sha256": "a" * 64,
        "source_receipt_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "requested_range": {
            "start": "2026-05-01",
            "end": "2026-05-02",
            "limit": None,
        },
    }

    assert not point_in_time_cache_covers_request(
        metadata,
        series_spec_sha256="a" * 64,
        source_identity=SOURCE_IDENTITY,
        start="2026-05-01T06:00:00Z",
        end="2026-05-01T12:00:00Z",
        limit=None,
    )


def test_point_in_time_cache_shortens_disk_name_without_shortening_identity(tmp_path):
    spec_hash = "a" * 64
    entry = point_in_time_cache_entry(
        adapter="abel",
        series_spec_sha256=spec_hash,
        cache_root=tmp_path,
    )

    assert entry.key == spec_hash
    assert entry.symbol == spec_hash
    assert entry.data_path.name == f"{'a' * 40}.csv"
    assert entry.meta_path.name == f"{'a' * 40}.json"


def test_concurrent_point_in_time_cache_writes_are_atomic(tmp_path, monkeypatch):
    point_spec = _point_spec()
    load_barrier = Barrier(2)
    write_state_lock = Lock()
    active_writes = 0
    peak_writes = 0

    class CanonicalModule:
        @staticmethod
        def load_cap_node_series(**kwargs):
            load_barrier.wait(timeout=5)
            frame = pd.DataFrame(
                {
                    "event_time": ["2024-01-02T05:00:00Z"],
                    "timestamp": ["2024-01-02T05:00:00Z"],
                    "value": [0.25],
                }
            )
            frame.attrs["source_receipt_sha256"] = "e" * 64
            frame.attrs["series_spec_sha256"] = point_spec.sha256
            return frame

    real_import = importlib.import_module

    def fake_import(name):
        if name == "abel_edge.plugins.abel.cap_node_series":
            return CanonicalModule()
        return real_import(name)

    real_to_csv = pd.DataFrame.to_csv

    def synchronized_to_csv(self, path_or_buf=None, *args, **kwargs):
        nonlocal active_writes, peak_writes
        if not str(path_or_buf).endswith(".partial"):
            return real_to_csv(self, path_or_buf, *args, **kwargs)
        with write_state_lock:
            active_writes += 1
            peak_writes = max(peak_writes, active_writes)
        try:
            sleep(0.05)
            return real_to_csv(self, path_or_buf, *args, **kwargs)
        finally:
            with write_state_lock:
                active_writes -= 1

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(pd.DataFrame, "to_csv", synchronized_to_csv)
    request = FeedLoadRequest(
        adapter="abel",
        kind="point_in_time_series",
        symbol=None,
        field=None,
        timeframe=None,
        start="2024-01-01",
        end="2024-01-31",
        limit=None,
        profile="daily",
        options={"cache_root": str(tmp_path)},
        strategy_id="demo",
        feed_name="graph_parent_01",
        series_spec=point_spec,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        frames = list(pool.map(lambda _: AbelDataFeedAdapter().load(request), range(2)))

    pd.testing.assert_frame_equal(frames[0], frames[1])
    assert peak_writes == 1
    cached = AbelDataFeedAdapter().load(request)
    pd.testing.assert_frame_equal(frames[0], cached)
