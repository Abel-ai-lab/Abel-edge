"""Visibility-order regressions for CAP canonical series."""

from __future__ import annotations

import pandas as pd
import pytest

from abel_edge.engine.feed_contract import DATE_GUARD_MODE_ENV, MAX_DATA_DATE_ENV
from abel_edge.engine.feed_loader import load_feed_frame
from abel_edge.plugins.abel.cap_node_series import (
    CanonicalNodeDataError,
    compile_cap_node_series_spec,
    load_cap_node_series,
    prepare_cap_node_series_spec,
)


NODE_ID = "health.openfda.drug.events:event_count#96bc3e82"
GRAPH_REF = {"graph_id": "abel-main", "graph_version": "CausalNodeV4"}


def _spec():
    return compile_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        source_receipt_sha256="e" * 64,
    )


def test_runtime_start_filters_availability_after_fetching_older_observations(
    monkeypatch,
):
    rows = [
        {
            "timestamp": "2026-01-02T00:00:00Z",
            "event_time": "2025-12-31T00:00:00Z",
            "node_id": NODE_ID,
            "value": 9.0,
        },
        {
            "timestamp": "2026-01-03T00:00:00Z",
            "event_time": "2026-01-03T00:00:00Z",
            "node_id": NODE_ID,
            "value": 10.0,
        },
    ]
    calls = []

    def fake_fetch_node_series(**kwargs):
        calls.append(kwargs)
        if kwargs["start"] is not None:
            return pd.DataFrame(rows[1:])
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        fake_fetch_node_series,
    )

    frame = load_cap_node_series(
        series_spec=_spec(),
        start="2026-01-01",
        end="2026-01-31",
        limit=None,
        config={},
    )

    assert calls[0]["start"] is None
    assert frame["value"].tolist() == [9.0, 10.0]


def test_runtime_limit_is_applied_after_availability_filtering(monkeypatch):
    rows = [
        {
            "timestamp": "2026-12-30T00:00:00Z",
            "event_time": "2026-12-30T00:00:00Z",
            "node_id": NODE_ID,
            "value": 9.0,
        },
        {
            "timestamp": "2027-01-02T00:00:00Z",
            "event_time": "2026-12-31T00:00:00Z",
            "node_id": NODE_ID,
            "value": 10.0,
        },
    ]
    calls = []

    def fake_fetch_node_series(**kwargs):
        calls.append(kwargs)
        if kwargs["limit"] is not None:
            return pd.DataFrame(rows[-int(kwargs["limit"]) :])
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        fake_fetch_node_series,
    )

    frame = load_cap_node_series(
        series_spec=_spec(),
        start=None,
        end="2026-12-31",
        limit=1,
        config={},
    )

    assert calls[0]["limit"] is None
    assert frame["value"].tolist() == [9.0]


@pytest.mark.parametrize("limit", [0, -1])
def test_runtime_limit_must_be_positive(limit):
    with pytest.raises(CanonicalNodeDataError, match="limit must be positive"):
        load_cap_node_series(
            series_spec=_spec(),
            start=None,
            end="2026-12-31",
            limit=limit,
            config={},
        )


def test_runtime_intraday_bounds_filter_exact_availability_times(monkeypatch):
    rows = [
        {
            "timestamp": "2026-01-01T01:00:00Z",
            "event_time": "2026-01-01T00:00:00Z",
            "node_id": NODE_ID,
            "value": 8.0,
        },
        {
            "timestamp": "2026-01-01T07:00:00Z",
            "event_time": "2026-01-01T00:00:00Z",
            "node_id": NODE_ID,
            "value": 9.0,
        },
        {
            "timestamp": "2026-01-01T12:00:00Z",
            "event_time": "2026-01-01T00:00:00Z",
            "node_id": NODE_ID,
            "value": 10.0,
        },
    ]
    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        lambda **kwargs: pd.DataFrame(rows),
    )

    frame = load_cap_node_series(
        series_spec=_spec(),
        start="2026-01-01T06:00:00Z",
        end="2026-01-01T08:00:00Z",
        limit=None,
        config={},
    )

    assert frame["value"].tolist() == [9.0]


def test_paper_prefetch_limit_is_applied_after_the_visible_end(monkeypatch):
    rows = [
        {
            "timestamp": "2026-01-01T12:00:00Z",
            "event_time": "2026-01-01T00:00:00Z",
            "node_id": NODE_ID,
            "value": 9.0,
        },
        {
            "timestamp": "2026-01-31T12:00:00Z",
            "event_time": "2026-01-31T00:00:00Z",
            "node_id": NODE_ID,
            "value": 10.0,
        },
    ]
    monkeypatch.setattr(
        "abel_edge.plugins.abel.cap_node_series.fetch_node_series",
        lambda **kwargs: pd.DataFrame(rows),
    )

    frame = load_feed_frame(
        {
            "name": "canonical",
            "kind": "point_in_time_series",
            "adapter": "abel",
            "series_spec": _spec().to_mapping(),
        },
        end="2026-01-01",
        request_end="2026-01-31",
        request_limit=1,
        limit=1,
    )

    assert frame["value"].tolist() == [9.0]


def test_future_availability_is_filtered_before_max_data_date_guard(monkeypatch):
    from abel_edge.plugins.abel.client import AbelClient

    monkeypatch.setenv("ABEL_API_KEY", "abel_test")
    monkeypatch.setenv(MAX_DATA_DATE_ENV, "2026-12-31")
    monkeypatch.setenv(DATE_GUARD_MODE_ENV, "fail-closed")
    rows = [
        {
            "date": "2026-12-30",
            "timestamp": "2026-12-30T12:00:00Z",
            "event_time": "2026-12-30T00:00:00Z",
            "node_id": NODE_ID,
            "value": 9.0,
        },
        {
            "date": "2026-12-31",
            "timestamp": "2027-01-02T12:00:00Z",
            "event_time": "2026-12-31T00:00:00Z",
            "node_id": NODE_ID,
            "value": 10.0,
        },
    ]

    monkeypatch.setattr(AbelClient, "fetch_node_series", lambda self, **kwargs: rows)

    frame = load_feed_frame(
        {
            "name": "canonical",
            "kind": "point_in_time_series",
            "adapter": "abel",
            "series_spec": _spec().to_mapping(),
        },
        end="2026-12-31",
    )

    assert frame["value"].tolist() == [9.0]


def test_preparation_receipt_excludes_rows_not_yet_visible_at_cutoff():
    rows = [
        {
            "timestamp": "2026-01-02T12:00:00Z",
            "event_time": "2025-12-31T00:00:00Z",
            "node_id": NODE_ID,
            "value": 9.0,
        },
        {
            "timestamp": "2027-01-02T12:00:00Z",
            "event_time": "2026-12-31T00:00:00Z",
            "node_id": NODE_ID,
            "value": 10.0,
        },
    ]
    calls = []

    def fake_fetcher(**kwargs):
        calls.append(kwargs)
        selected = rows[1:] if kwargs["start"] is not None else rows
        if kwargs["limit"] is not None:
            selected = selected[-int(kwargs["limit"]) :]
        return pd.DataFrame(selected)

    spec = prepare_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        start="2026-01-01",
        end="2026-12-31",
        limit=1,
        fetcher=fake_fetcher,
    )

    assert calls[0]["start"] is None
    assert calls[0]["limit"] is None
    assert spec.payload["provenance"]["source_observation_count"] == 1
    assert (
        spec.payload["provenance"]["source_last_timestamp"]
        == "2026-01-02T12:00:00Z"
    )


def test_preparation_uses_global_cutoff_as_open_ended_visibility_bound(monkeypatch):
    monkeypatch.setenv(MAX_DATA_DATE_ENV, "2026-01-02")
    monkeypatch.setenv(DATE_GUARD_MODE_ENV, "fail-closed")
    calls = []

    def fake_fetcher(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "timestamp": "2026-01-02T12:00:00Z",
                    "event_time": "2026-01-01T00:00:00Z",
                    "node_id": NODE_ID,
                    "value": 9.0,
                },
                {
                    "timestamp": "2026-01-03T00:00:00Z",
                    "event_time": "2026-01-02T00:00:00Z",
                    "node_id": NODE_ID,
                    "value": 10.0,
                },
            ]
        )

    spec = prepare_cap_node_series_spec(
        node_id=NODE_ID,
        graph_ref=GRAPH_REF,
        start=None,
        end=None,
        fetcher=fake_fetcher,
    )

    assert calls[0]["end"] == "2026-01-02"
    assert spec.payload["provenance"]["source_observation_count"] == 1
    assert (
        spec.payload["provenance"]["source_last_timestamp"]
        == "2026-01-02T12:00:00Z"
    )


def test_preparation_limit_must_be_positive():
    with pytest.raises(CanonicalNodeDataError, match="limit must be positive"):
        prepare_cap_node_series_spec(
            node_id=NODE_ID,
            graph_ref=GRAPH_REF,
            start="2026-01-01",
            end="2026-12-31",
            limit=0,
            fetcher=lambda **kwargs: pd.DataFrame(),
        )
