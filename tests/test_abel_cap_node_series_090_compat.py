"""Compatibility checks for live execution of 0.9.0 canonical specs."""

from __future__ import annotations

import importlib
from dataclasses import replace

import pandas as pd

from abel_edge.engine.adapter_registry import AbelDataFeedAdapter, FeedLoadRequest
from abel_edge.engine.feed_loader import load_feed_frame
from abel_edge.plugins.abel.cap_node_series import (
    cap_node_series_receipt,
    compile_cap_node_series_spec,
)


NODE_ID = "macro.example.release"
GRAPH_REF = {"graph_id": "graph-v4", "graph_version": "2026-06-01"}
PREPARED_ROW = {
    "timestamp": "2026-06-30T12:00:00Z",
    "event_time": "2026-06-30T00:00:00Z",
    "node_id": NODE_ID,
    "value": 10.0,
}
APPENDED_ROW = {
    "timestamp": "2026-07-02T12:00:00Z",
    "event_time": "2026-07-02T00:00:00Z",
    "node_id": NODE_ID,
    "value": 11.0,
}


def _old_spec():
    return compile_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        source_receipt_sha256=cap_node_series_receipt(
            [PREPARED_ROW],
            node_id=NODE_ID,
        ),
    )


def _old_feed(spec, **extra):
    return {
        "name": "graph_node_old_alias",
        "kind": "point_in_time_series",
        "adapter": "abel",
        "profile": "daily",
        "series_spec": spec.to_mapping(),
        "source_start": "2026-06-01",
        "source_end": "2026-06-30",
        "source_limit": 30,
        **extra,
    }


def test_old_spec_uses_explicit_runtime_end_instead_of_prepared_end(monkeypatch):
    calls = []
    rows = [
        PREPARED_ROW,
        APPENDED_ROW,
        {
            "timestamp": "2026-07-06T12:00:00Z",
            "event_time": "2026-07-01T00:00:00Z",
            "node_id": NODE_ID,
            "value": 99.0,
        },
    ]

    def fake_fetch_node_series(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        fake_fetch_node_series,
    )
    spec = _old_spec()

    result = load_feed_frame(
        _old_feed(spec),
        start="2026-06-01",
        end="2026-07-05",
    )

    assert calls[0]["start"] is None
    assert calls[0]["end"] == "2026-07-05"
    assert calls[0]["limit"] is None
    assert result["timestamp"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-06-30",
        "2026-07-02",
    ]
    assert (
        result.attrs["source_receipt_sha256"]
        != spec.payload["provenance"]["source_receipt_sha256"]
    )


def test_old_source_window_options_do_not_reach_live_materialization(monkeypatch):
    spec = _old_spec()
    calls = []

    class CanonicalModule:
        @staticmethod
        def load_cap_node_series(**kwargs):
            calls.append(kwargs)
            frame = pd.DataFrame([PREPARED_ROW]).drop(columns=["node_id"])
            frame.attrs["source_receipt_sha256"] = cap_node_series_receipt(
                [PREPARED_ROW],
                node_id=NODE_ID,
            )
            frame.attrs["series_spec_sha256"] = spec.sha256
            return frame

    real_import = importlib.import_module

    def fake_import(name):
        if name == "abel_edge.plugins.abel.cap_node_series":
            return CanonicalModule()
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    load_feed_frame(_old_feed(spec), end="2026-06-30")

    assert calls[0]["config"] == {"env_path": ".env"}


def test_old_spec_open_ended_requests_do_not_reuse_a_snapshot(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_node_series(**kwargs):
        calls.append(kwargs)
        rows = [PREPARED_ROW] if len(calls) == 1 else [PREPARED_ROW, APPENDED_ROW]
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        fake_fetch_node_series,
    )
    spec = _old_spec()
    feed = _old_feed(spec, cache_root=str(tmp_path))

    first = load_feed_frame(feed, start="2026-06-01", end=None)
    second = load_feed_frame(feed, start="2026-06-01", end=None)

    assert len(calls) == 2
    assert calls[0]["end"] is None
    assert len(first) == 1
    assert second["value"].tolist() == [10.0, 11.0]


def test_explicit_runtime_end_refreshes_then_reuses_cache(tmp_path, monkeypatch):
    spec = _old_spec()
    calls = []

    class CanonicalModule:
        @staticmethod
        def load_cap_node_series(**kwargs):
            calls.append(kwargs["end"])
            rows = (
                [PREPARED_ROW] if kwargs["end"] == "2026-06-30" else [PREPARED_ROW, APPENDED_ROW]
            )
            frame = pd.DataFrame(rows).drop(columns=["node_id"])
            frame.attrs["source_receipt_sha256"] = cap_node_series_receipt(
                rows,
                node_id=NODE_ID,
            )
            frame.attrs["series_spec_sha256"] = spec.sha256
            return frame

    real_import = importlib.import_module

    def fake_import(name):
        if name == "abel_edge.plugins.abel.cap_node_series":
            return CanonicalModule()
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    initial_request = FeedLoadRequest(
        adapter="abel",
        kind="point_in_time_series",
        symbol=None,
        field=None,
        timeframe=None,
        start="2026-06-01",
        end="2026-06-30",
        limit=None,
        profile="daily",
        options={"cache_root": str(tmp_path)},
        strategy_id="old-v4",
        feed_name="graph_node_old_alias",
        series_spec=spec,
    )
    advanced_request = replace(initial_request, end="2026-07-05")
    adapter = AbelDataFeedAdapter()

    adapter.load(initial_request)
    advanced = adapter.load(advanced_request)
    repeated = adapter.load(advanced_request)

    assert calls == ["2026-06-30", "2026-07-05"]
    assert advanced["value"].tolist() == [10.0, 11.0]
    assert repeated["value"].tolist() == [10.0, 11.0]
