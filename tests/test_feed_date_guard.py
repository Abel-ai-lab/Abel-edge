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
