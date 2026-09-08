"""Minimal CAP scalar-node compilation and materialization contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from abel_edge.engine.point_in_time_series import PointInTimeSeriesSpec
from abel_edge.plugins.abel import (
    compile_cap_node_series_spec,
    prepare_cap_node_series_spec,
)
from abel_edge.plugins.abel.cap_node_series import (
    CanonicalNodeDataError,
    cap_node_series_receipt,
    load_cap_node_series,
)


NODE_ID = "health.openfda.drug.events:event_count#96bc3e82"
GRAPH_REF = {"graph_id": "abel-main", "graph_version": "CausalNodeV4"}
ROWS = [
    {
        "timestamp": "2026-05-01T00:00:00Z",
        "event_time": "2026-04-30T00:00:00Z",
        "node_id": NODE_ID,
        "value": 21.0,
    },
    {
        "timestamp": "2026-05-02T00:00:00Z",
        "node_id": NODE_ID,
        "value": 25.0,
    },
]


def test_cap_node_series_compiles_to_asof_point_in_time_feed():
    receipt = cap_node_series_receipt(ROWS, node_id=NODE_ID)

    spec = compile_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        source_receipt_sha256=receipt,
    )

    payload = spec.payload
    assert payload["series_id"] == NODE_ID
    assert payload["source"] == {
        "adapter": "abel",
        "request": {
            "node_id": NODE_ID,
            "retrieval_mode": "node_series",
            "graph_ref": GRAPH_REF,
        },
    }
    assert payload["schema"] == {
        "event_time_field": "event_time",
        "available_at_field": "timestamp",
        "value_field": "value",
    }
    assert payload["materialization"] == {
        "frequency": "irregular",
        "timezone": "UTC",
        "missing_policy": "none",
        "alignment_policy": "asof",
    }
    assert payload["availability"] == {"mode": "explicit"}
    assert payload["transforms"] == []


def test_cap_node_series_materializer_loads_exact_series_and_receipt(monkeypatch):
    receipt = cap_node_series_receipt(ROWS, node_id=NODE_ID)
    spec = compile_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        source_receipt_sha256=receipt,
    )
    calls = []

    def fake_fetch_node_series(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(ROWS)

    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        fake_fetch_node_series,
    )

    frame = load_cap_node_series(
        series_spec=spec,
        start="2026-05-01",
        end="2026-05-02",
        limit=20,
        config={},
    )

    assert list(frame.columns) == ["event_time", "timestamp", "value"]
    assert list(frame["event_time"]) == [
        "2026-04-30T00:00:00Z",
        "2026-05-02T00:00:00Z",
    ]
    assert list(frame["timestamp"]) == [
        "2026-05-01T00:00:00Z",
        "2026-05-02T00:00:00Z",
    ]
    assert list(frame["value"]) == [21.0, 25.0]
    assert frame.attrs["source_receipt_sha256"] == receipt
    assert frame.attrs["series_spec_sha256"] == spec.sha256
    assert calls == [
        {
            "node_id": NODE_ID,
            "start": "2026-05-01",
            "end": "2026-05-02",
            "limit": 20,
            "config": {},
        }
    ]


def test_cap_runtime_window_uses_caller_bounds(monkeypatch):
    rows = [
        {
            "timestamp": "2026-01-02T00:00:00Z",
            "event_time": "2026-01-01T00:00:00Z",
            "node_id": NODE_ID,
            "value": 10.0,
        },
        *ROWS,
        {
            "timestamp": "2026-12-02T00:00:00Z",
            "event_time": "2026-12-01T00:00:00Z",
            "node_id": NODE_ID,
            "value": 30.0,
        },
    ]
    calls = []

    def fake_fetch_node_series(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        fake_fetch_node_series,
    )
    spec = compile_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        source_receipt_sha256=cap_node_series_receipt(rows, node_id=NODE_ID),
    )

    frame = load_cap_node_series(
        series_spec=spec,
        start="2026-05-01",
        end="2026-05-31",
        limit=1,
        config={"source_start": "2026-01-01", "source_end": "2026-12-31"},
    )

    assert (calls[0]["start"], calls[0]["end"], calls[0]["limit"]) == (
        "2026-05-01",
        "2026-05-31",
        1,
    )
    assert list(frame["timestamp"]) == ["2026-05-02T00:00:00Z"]
    assert list(frame["value"]) == [25.0]


def test_cap_node_series_materializer_records_response_receipt_drift(monkeypatch):
    spec = compile_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        source_receipt_sha256="e" * 64,
    )
    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        lambda **_: pd.DataFrame(ROWS),
    )

    frame = load_cap_node_series(
        series_spec=spec,
        start="2026-05-01",
        end="2026-05-02",
        limit=20,
        config={},
    )

    assert frame.attrs["source_receipt_sha256"] == cap_node_series_receipt(
        ROWS,
        node_id=NODE_ID,
    )


def test_cap_node_series_rejects_spec_request_node_mismatch():
    payload = compile_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        source_receipt_sha256="e" * 64,
    ).payload
    payload["source"]["request"]["node_id"] = "another.node"

    with pytest.raises(CanonicalNodeDataError, match="node_id.*series_spec.series_id"):
        load_cap_node_series(
            series_spec=PointInTimeSeriesSpec.from_mapping(payload),
            start="2026-05-01",
            end="2026-05-02",
            limit=None,
            config={},
        )


def test_cap_node_series_materializer_rejects_unapplied_transforms(monkeypatch):
    payload = compile_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        source_receipt_sha256=cap_node_series_receipt(ROWS, node_id=NODE_ID),
    ).payload
    payload["transforms"] = [{"op": "scale", "parameters": {"factor": 2.0}}]
    spec = PointInTimeSeriesSpec.from_mapping(payload)
    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        lambda **_: pd.DataFrame(ROWS),
    )

    with pytest.raises(CanonicalNodeDataError, match="does not support series_spec.transforms"):
        load_cap_node_series(
            series_spec=spec,
            start="2026-05-01",
            end="2026-05-02",
            limit=20,
            config={},
        )


def test_prepare_cap_node_series_spec_records_live_response_receipt(monkeypatch):
    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        lambda **_: pd.DataFrame(ROWS),
    )

    spec = prepare_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        start="2026-05-01",
        end="2026-05-02",
        limit=20,
        config={},
    )

    assert spec.payload["provenance"]["source_receipt_sha256"] == (
        cap_node_series_receipt(ROWS, node_id=NODE_ID)
    )
    assert spec.payload["source"]["request"]["node_id"] == NODE_ID
    assert spec.payload["provenance"]["source_observation_count"] == 2
    assert (
        spec.payload["provenance"]["source_first_timestamp"]
        == "2026-05-01T00:00:00Z"
    )
    assert (
        spec.payload["provenance"]["source_last_timestamp"]
        == "2026-05-02T00:00:00Z"
    )


def test_prepare_cap_node_series_spec_rejects_empty_live_series(monkeypatch):
    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        lambda **_: pd.DataFrame(columns=["timestamp", "node_id", "value"]),
    )

    with pytest.raises(CanonicalNodeDataError, match="returned no observations"):
        prepare_cap_node_series_spec(
            node_id=NODE_ID,
            graph_ref=GRAPH_REF,
            start="2026-05-01",
            end="2026-05-02",
            limit=20,
            config={},
        )
