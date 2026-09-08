"""Behavior tests for generic point-in-time auxiliary series."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from abel_edge.engine.point_in_time_series import (
    PointInTimeSeriesContractError,
    PointInTimeSeriesSpec,
    assert_point_in_time_adapter_identity,
    normalize_point_in_time_series_frame,
)


def _series_spec(*, path: str = "macro.csv") -> dict:
    source_receipt = "a" * 64
    source_path = Path(path)
    if source_path.is_file():
        source_receipt = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return {
        "contract": "abel-edge.point-in-time-series/v1",
        "series_id": "macro.cpi.us",
        "source": {
            "adapter": "csv",
            "request": {"path": path},
        },
        "schema": {
            "event_time_field": "observed_at",
            "available_at_field": "released_at",
            "value_field": "reading",
        },
        "materialization": {
            "frequency": "irregular",
            "timezone": "UTC",
            "missing_policy": "none",
            "alignment_policy": "asof",
        },
        "transforms": [],
        "availability": {"mode": "explicit"},
        "provenance": {
            "source_receipt_sha256": source_receipt,
            "schema_sha256": "b" * 64,
        },
    }


def test_spec_hash_is_canonical_and_rejects_embedded_credentials():
    payload = _series_spec()
    reordered = {key: payload[key] for key in reversed(payload)}

    first = PointInTimeSeriesSpec.from_mapping(payload)
    second = PointInTimeSeriesSpec.from_mapping(reordered)

    assert first.sha256 == second.sha256
    assert first.series_id == "macro.cpi.us"
    assert first.source_adapter == "csv"

    unsafe = deepcopy(payload)
    unsafe["source"]["request"]["api_key"] = "must-not-enter-a-spec"
    with pytest.raises(PointInTimeSeriesContractError, match="credential"):
        PointInTimeSeriesSpec.from_mapping(unsafe)


def test_spec_allows_domain_token_fields_but_rejects_explicit_auth_tokens():
    public_request = _series_spec()
    public_request["source"]["request"].update(
        {
            "token_address": "0x1234",
            "token_symbol": "ABEL",
        }
    )

    spec = PointInTimeSeriesSpec.from_mapping(public_request)

    assert spec.source_request["token_address"] == "0x1234"
    assert spec.source_request["token_symbol"] == "ABEL"

    unsafe = deepcopy(public_request)
    unsafe["source"]["request"]["access_token"] = "must-stay-runtime-only"
    with pytest.raises(PointInTimeSeriesContractError, match="credential"):
        PointInTimeSeriesSpec.from_mapping(unsafe)

    provider_prefixed = deepcopy(public_request)
    provider_prefixed["source"]["request"]["cap_api_key"] = "must-stay-runtime-only"
    with pytest.raises(PointInTimeSeriesContractError, match="credential"):
        PointInTimeSeriesSpec.from_mapping(provider_prefixed)


def test_spec_rejects_non_finite_numbers_before_hashing():
    payload = _series_spec()
    payload["transforms"] = [
        {
            "op": "scale",
            "parameters": {"alpha": float("nan")},
        }
    ]

    with pytest.raises(PointInTimeSeriesContractError, match="finite JSON"):
        PointInTimeSeriesSpec.from_mapping(payload)


def test_spec_rejects_noncanonical_receipt_case_and_grid_time():
    uppercase_receipt = _series_spec()
    uppercase_receipt["provenance"]["source_receipt_sha256"] = "A" * 64
    with pytest.raises(PointInTimeSeriesContractError, match="lowercase SHA-256"):
        PointInTimeSeriesSpec.from_mapping(uppercase_receipt)

    loose_grid = _series_spec()
    loose_grid["materialization"] = {
        "frequency": "calendar_day",
        "timezone": "UTC",
        "grid_time_utc": "5 hours",
        "missing_policy": "none",
        "alignment_policy": "native_only",
    }
    with pytest.raises(PointInTimeSeriesContractError, match="HH:MM:SS"):
        PointInTimeSeriesSpec.from_mapping(loose_grid)


def test_normalization_uses_available_at_as_strategy_visible_timestamp():
    spec = PointInTimeSeriesSpec.from_mapping(_series_spec())
    raw = pd.DataFrame(
        {
            "observed_at": [
                "2024-01-02T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "released_at": [
                "2024-01-03T05:00:00Z",
                "2024-01-02T05:00:00Z",
            ],
            "reading": [101.0, 100.0],
        }
    )

    frame = normalize_point_in_time_series_frame(raw, spec, name="macro")

    assert list(frame.columns) == ["timestamp", "event_time", "available_at", "value"]
    assert list(frame["value"]) == [100.0, 101.0]
    assert list(frame["timestamp"]) == list(frame["available_at"])
    assert frame.iloc[0]["timestamp"] == pd.Timestamp("2024-01-02T05:00:00Z")
    assert frame.attrs["series_id"] == "macro.cpi.us"
    assert frame.attrs["series_spec_sha256"] == spec.sha256


def test_adapter_identity_requires_actual_receipt_without_comparing_contents():
    spec = PointInTimeSeriesSpec.from_mapping(_series_spec())
    raw = pd.DataFrame()
    raw.attrs["series_spec_sha256"] = spec.sha256

    with pytest.raises(PointInTimeSeriesContractError, match="did not return a source receipt"):
        assert_point_in_time_adapter_identity(
            raw,
            spec,
            name="live canonical",
            verify_source_receipt=False,
        )


def test_normalization_does_not_align_values_by_adapter_index_labels():
    spec = PointInTimeSeriesSpec.from_mapping(_series_spec())
    raw = pd.DataFrame(
        {
            "observed_at": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
            ],
            "released_at": [
                "2024-01-02T05:00:00Z",
                "2024-01-03T05:00:00Z",
            ],
            "reading": [100.0, 101.0],
        },
        index=[10, 20],
    )

    frame = normalize_point_in_time_series_frame(raw, spec, name="macro")

    assert list(frame["value"]) == [100.0, 101.0]
    assert list(frame["available_at"]) == [
        pd.Timestamp("2024-01-02T05:00:00Z"),
        pd.Timestamp("2024-01-03T05:00:00Z"),
    ]


def test_normalization_accepts_an_empty_frame_with_stable_utc_schema():
    spec = PointInTimeSeriesSpec.from_mapping(_series_spec())

    frame = normalize_point_in_time_series_frame(
        pd.DataFrame(columns=["observed_at", "released_at", "reading"]),
        spec,
        name="empty macro",
    )

    assert frame.empty
    assert list(frame.columns) == ["timestamp", "event_time", "available_at", "value"]
    assert isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(frame["timestamp"].dtype.tz) == "UTC"


def test_normalization_rejects_duplicate_visibility_times_and_non_finite_values():
    spec = PointInTimeSeriesSpec.from_mapping(_series_spec())
    duplicate = pd.DataFrame(
        {
            "observed_at": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "released_at": ["2024-01-03T05:00:00Z", "2024-01-03T05:00:00Z"],
            "reading": [100.0, 101.0],
        }
    )
    with pytest.raises(PointInTimeSeriesContractError, match="duplicate available_at"):
        normalize_point_in_time_series_frame(duplicate, spec, name="macro")

    non_finite = duplicate.iloc[:1].copy()
    non_finite["reading"] = [float("inf")]
    with pytest.raises(PointInTimeSeriesContractError, match="finite"):
        normalize_point_in_time_series_frame(non_finite, spec, name="macro")


def test_calendar_day_series_applies_availability_lag_after_grid_alignment():
    payload = _series_spec()
    payload["schema"].pop("available_at_field")
    payload["materialization"] = {
        "frequency": "calendar_day",
        "timezone": "UTC",
        "grid_time_utc": "05:00:00",
        "missing_policy": "none",
        "alignment_policy": "native_only",
    }
    payload["availability"] = {"mode": "calendar_days", "lag_days": 2}
    spec = PointInTimeSeriesSpec.from_mapping(payload)
    raw = pd.DataFrame(
        {
            "observed_at": ["2024-01-01T05:00:00Z"],
            "reading": [100.0],
        }
    )

    frame = normalize_point_in_time_series_frame(raw, spec, name="graph node")

    assert frame.iloc[0]["event_time"] == pd.Timestamp("2024-01-01T05:00:00Z")
    assert frame.iloc[0]["available_at"] == pd.Timestamp("2024-01-03T05:00:00Z")
    assert frame.iloc[0]["timestamp"] == frame.iloc[0]["available_at"]


def test_calendar_grid_rejects_rows_at_the_wrong_utc_time():
    payload = _series_spec()
    payload["schema"].pop("available_at_field")
    payload["materialization"] = {
        "frequency": "calendar_day",
        "timezone": "UTC",
        "grid_time_utc": "05:00:00",
        "missing_policy": "none",
        "alignment_policy": "native_only",
    }
    payload["availability"] = {"mode": "calendar_days", "lag_days": 1}
    spec = PointInTimeSeriesSpec.from_mapping(payload)

    with pytest.raises(PointInTimeSeriesContractError, match="grid_time_utc"):
        normalize_point_in_time_series_frame(
            pd.DataFrame(
                {
                    "observed_at": ["2024-01-01T00:00:00Z"],
                    "reading": [100.0],
                }
            ),
            spec,
            name="graph node",
        )
