"""Runtime execution tests for the data-feed contract."""

from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

from abel_edge.cli import main
from abel_edge.engine.ledger import read_trade_log
from tests.data_contract_helpers import (
    BAR_FEED_ENGINE_CODE,
    FEED_ENGINE_CODE,
    NAIVE_ENGINE_CODE,
    PRIMARY_ONLY_ENGINE_CODE,
    UNDECLARED_FEED_ENGINE_CODE,
    reset_strategy_modules,
    write_engine_project,
)


def test_run_fails_early_on_naive_signal_dates(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_engine_project(root, engine_name="naive_dates", engine_code=NAIVE_ENGINE_CODE)
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "naive_dates"])

        assert result.exit_code != 0
        assert "NaiveDatesEngine" in result.output
        assert "UTC-aware" in result.output
        sys.path.pop(0)
        reset_strategy_modules()


def test_paper_fails_early_on_naive_signal_dates(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_engine_project(root, engine_name="naive_dates", engine_code=NAIVE_ENGINE_CODE)
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["paper", "--strategy", "naive_dates"])

        assert result.exit_code != 0
        assert "NaiveDatesEngine" in result.output
        assert "UTC-aware" in result.output
        sys.path.pop(0)
        reset_strategy_modules()


def test_run_supports_declared_series_feed_via_framework_path(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        (root / "data").mkdir(exist_ok=True)
        (root / "data" / "risk_scale.csv").write_text(
            "timestamp,value\n"
            "2026-01-01T00:00:00Z,0.2\n"
            "2026-01-02T00:00:00Z,0.4\n"
            "2026-01-03T00:00:00Z,0.6\n",
            encoding="utf-8",
        )
        extra_yaml = """
    feeds:
      risk_scale:
        kind: series
        source: csv
        path: data/risk_scale.csv
        field: value
"""
        write_engine_project(root, engine_name="feed_demo", engine_code=FEED_ENGINE_CODE, extra_yaml=extra_yaml)
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "feed_demo"])

        assert result.exit_code == 0, result.output
        trade_df = read_trade_log("data/trade_log_feed_demo.csv")
        assert list(trade_df["position"].round(2)) == [0.2, 0.4, 0.6]
        sys.path.pop(0)
        reset_strategy_modules()


def test_run_supports_declared_series_feed_with_naive_csv_timestamps(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        (root / "data").mkdir(exist_ok=True)
        (root / "data" / "risk_scale.csv").write_text(
            "timestamp,value\n"
            "2026-01-01,0.2\n"
            "2026-01-02,0.4\n"
            "2026-01-03,0.6\n",
            encoding="utf-8",
        )
        extra_yaml = """
    feeds:
      risk_scale:
        kind: series
        source: csv
        path: data/risk_scale.csv
        field: value
"""
        write_engine_project(
            root,
            engine_name="feed_demo",
            engine_code=FEED_ENGINE_CODE,
            extra_yaml=extra_yaml,
        )
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "feed_demo"])

        assert result.exit_code == 0, result.output
        trade_df = read_trade_log("data/trade_log_feed_demo.csv")
        assert list(trade_df["position"].round(2)) == [0.2, 0.4, 0.6]
        sys.path.pop(0)
        reset_strategy_modules()


def test_run_supports_naive_primary_feed_timestamps_from_csv_loader(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_engine_project(
            root,
            engine_name="primary_only",
            engine_code=PRIMARY_ONLY_ENGINE_CODE,
            primary_csv=(
                "timestamp,close\n"
                "2026-01-01,100\n"
                "2026-01-02,110\n"
                "2026-01-03,120\n"
            ),
        )
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "primary_only"])

        assert result.exit_code == 0, result.output
        trade_df = read_trade_log("data/trade_log_primary_only.csv")
        assert list(trade_df["date"].dt.strftime("%Y-%m-%d")) == [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ]
        sys.path.pop(0)
        reset_strategy_modules()


def test_run_supports_declared_bars_feed_via_framework_path(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        (root / "data").mkdir(exist_ok=True)
        (root / "data" / "btcusd.csv").write_text(
            "timestamp,close\n"
            "2026-01-01T00:00:00Z,20\n"
            "2026-01-02T00:00:00Z,40\n"
            "2026-01-03T00:00:00Z,60\n",
            encoding="utf-8",
        )
        extra_yaml = """
    feeds:
      btc_ref:
        kind: bars
        source: csv
        path: data/btcusd.csv
        symbol: BTCUSD
"""
        write_engine_project(
            root,
            engine_name="bars_feed_demo",
            engine_code=BAR_FEED_ENGINE_CODE,
            extra_yaml=extra_yaml,
        )
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "bars_feed_demo"])

        assert result.exit_code == 0, result.output
        trade_df = read_trade_log("data/trade_log_bars_feed_demo.csv")
        assert list(trade_df["position"].round(2)) == [0.2, 0.4, 0.6]
        sys.path.pop(0)
        reset_strategy_modules()


def test_run_fails_early_on_undeclared_feed_access(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_engine_project(
            root,
            engine_name="undeclared_feed",
            engine_code=UNDECLARED_FEED_ENGINE_CODE,
        )
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "undeclared_feed"])

        assert result.exit_code != 0
        assert "UndeclaredFeedEngine" in result.output
        assert "not declared" in result.output
        sys.path.pop(0)
        reset_strategy_modules()


def test_paper_fails_early_on_undeclared_feed_access(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_engine_project(
            root,
            engine_name="undeclared_feed",
            engine_code=UNDECLARED_FEED_ENGINE_CODE,
        )
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["paper", "--strategy", "undeclared_feed"])

        assert result.exit_code != 0
        assert "UndeclaredFeedEngine" in result.output
        assert "not declared" in result.output
        sys.path.pop(0)
        reset_strategy_modules()
