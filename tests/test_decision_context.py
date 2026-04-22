"""Unit tests for the agent-first decision context runtime."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_edge.engine.base import StrategyEngine


def _write_feed_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    path.write_text(
        "timestamp,close\n"
        + "\n".join(f"{timestamp},{close}" for timestamp, close in rows)
        + "\n",
        encoding="utf-8",
    )


def _write_feed_with_volume_csv(path: Path, rows: list[tuple[str, float, float]]) -> None:
    path.write_text(
        "timestamp,close,volume\n"
        + "\n".join(f"{timestamp},{close},{volume}" for timestamp, close, volume in rows)
        + "\n",
        encoding="utf-8",
    )


def _context_with_csv_feeds(tmp_path: Path) -> dict:
    primary_path = tmp_path / "primary.csv"
    driver_path = tmp_path / "driver.csv"
    _write_feed_csv(
        primary_path,
        [
            ("2024-01-01T00:00:00Z", 100.0),
            ("2024-01-02T00:00:00Z", 102.0),
            ("2024-01-03T00:00:00Z", 101.0),
            ("2024-01-04T00:00:00Z", 104.0),
            ("2024-01-05T00:00:00Z", 106.0),
        ],
    )
    _write_feed_csv(
        driver_path,
        [
            ("2024-01-01T00:00:00Z", 95.0),
            ("2024-01-02T00:00:00Z", 96.0),
            ("2024-01-03T00:00:00Z", 97.0),
            ("2024-01-04T00:00:00Z", 98.0),
            ("2024-01-05T00:00:00Z", 99.0),
        ],
    )
    return {
        "_data_contract": {"profile": "daily"},
        "_runtime_profile": {
            "profile": "daily",
            "target": "AAA",
            "decision_event": "bar_close",
            "execution_delay_bars": 1,
            "return_basis": "close_to_close",
        },
        "_execution_constraints": {},
        "_feeds": {
            "primary": {
                "name": "primary",
                "kind": "bars",
                "adapter": "csv",
                "symbol": "AAA",
                "timeframe": "1d",
                "profile": "daily",
                "path": str(primary_path),
            },
            "driver": {
                "name": "driver",
                "kind": "bars",
                "adapter": "csv",
                "symbol": "BBB",
                "timeframe": "1d",
                "profile": "daily",
                "path": str(driver_path),
            },
        },
    }


def test_batch_view_supports_feed_asof_and_inspection(tmp_path):
    class BatchEngine(StrategyEngine):
        def compute_decisions(self, ctx):
            assert ctx.available_feeds() == ["driver", "primary"]
            inspect = ctx.inspect_feed("driver")
            assert inspect["rows"] == 5
            close = ctx.target.series("close")
            driver = ctx.feed("driver").asof_series("close")
            next_position = (close > driver).astype(float)
            return ctx.decisions(next_position)

    engine = BatchEngine(context=_context_with_csv_feeds(tmp_path))
    compiled = engine.compute_runtime_output()

    assert list(compiled.next_position.round(2)) == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert list(compiled.positions.round(2)) == [0.0, 1.0, 1.0, 1.0, 1.0]
    surfaces = {item["surface"] for item in engine.latest_decision_trace()}
    assert "target.series" in surfaces
    assert "feed.asof_series" in surfaces


def test_point_view_supports_history_between_and_trace_point(tmp_path):
    class PointEngine(StrategyEngine):
        def compute_decisions(self, ctx):
            assert ctx.trace_point("2024-01-03T00:00:00Z")["target_close"] == 101.0
            next_position = []
            for point in ctx.points():
                target_window = point.target.history("close", bars=2)
                driver_window = point.feed("driver").between(
                    point.prev_target_close(),
                    point.decision_time(),
                    field="close",
                )
                signal = float(len(target_window) == 2 and len(driver_window) >= 1)
                next_position.append(signal)
            samples = ctx.sample_points(limit=3)
            assert len(samples) == 3
            return ctx.decisions(next_position)

    engine = PointEngine(context=_context_with_csv_feeds(tmp_path))
    compiled = engine.compute_runtime_output()

    assert list(compiled.next_position.round(2)) == [0.0, 1.0, 1.0, 1.0, 1.0]
    surfaces = {item["surface"] for item in engine.latest_decision_trace()}
    assert "point.target.history" in surfaces
    assert "point.feed.between" in surfaces


def test_feed_default_field_can_follow_volume_nodes(tmp_path):
    primary_path = tmp_path / "primary.csv"
    driver_path = tmp_path / "driver.csv"
    _write_feed_csv(
        primary_path,
        [
            ("2024-01-01T00:00:00Z", 100.0),
            ("2024-01-02T00:00:00Z", 102.0),
        ],
    )
    _write_feed_with_volume_csv(
        driver_path,
        [
            ("2024-01-01T00:00:00Z", 95.0, 1000.0),
            ("2024-01-02T00:00:00Z", 96.0, 1200.0),
        ],
    )

    class VolumeEngine(StrategyEngine):
        def compute_decisions(self, ctx):
            volume = ctx.feed("driver").asof_series()
            assert list(volume.astype(float)) == [1000.0, 1200.0]
            return ctx.decisions([0.0, 0.0])

    engine = VolumeEngine(
        context={
            "_runtime_profile": {"profile": "daily", "target": "AAA", "target_node": "AAA.price"},
            "_feeds": {
                "primary": {
                    "name": "primary",
                    "kind": "bars",
                    "adapter": "csv",
                    "symbol": "AAA",
                    "timeframe": "1d",
                    "profile": "daily",
                    "path": str(primary_path),
                },
                "driver": {
                    "name": "driver",
                    "kind": "bars",
                    "adapter": "csv",
                    "symbol": "AAA",
                    "timeframe": "1d",
                    "profile": "daily",
                    "path": str(driver_path),
                    "default_field": "volume",
                },
            },
        }
    )

    compiled = engine.compute_runtime_output()

    assert list(compiled.next_position.round(2)) == [0.0, 0.0]
