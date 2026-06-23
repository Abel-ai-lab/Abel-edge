from pathlib import Path
import importlib
import sys

from click.testing import CliRunner
import pytest

from abel_edge.config import load_config
from abel_edge.engine.ledger import read_trade_log
from abel_edge.engine.trader import SYSTEM_LOOKBACK_PADDING_BARS, paper_run_one
from tests.paper_once_fixtures import (
    BOOTSTRAP_CONTEXT_ENGINE_CODE,
    DECISION_ENGINE_CODE,
    DIRECT_SIGNAL_ENGINE_CODE,
    write_bootstrap_log,
    write_project,
)


def test_paper_run_one_default_signal_computes_once_and_rolls_position(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_project(root, package_name="decision_once", engine_code=DECISION_ENGINE_CODE, days=4)
        sys.path.insert(0, str(root))

        try:
            write_bootstrap_log(
                "data/trade_log_decision_once.csv",
                dates=["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
                closes=[100.0, 110.0],
                positions=[0.0, 0.0],
                next_positions=[0.0, 0.25],
            )
            cfg = load_config()

            result = paper_run_one(
                cfg["strategies"][0],
                settings=cfg.get("settings"),
                as_of="2026-01-04T00:00:00Z",
            )

            engine_module = importlib.import_module("strategies.decision_once.engine")
            assert len(engine_module.CountingDecisionEngine.calls) == 1
            assert result["execution_mode"] == "compiled_output"
            assert result["paper_history_boundary"]["source"] == "legacy_default"
            assert result["n_rows"] == 2
            paper_df = read_trade_log("data/paper_log_decision_once.csv")
            assert list(paper_df["date"].dt.strftime("%Y-%m-%d")) == [
                "2026-01-03",
                "2026-01-04",
            ]
            assert list(paper_df["position"].round(2)) == [0.25, 0.50]
            assert list(paper_df["next_position"].round(2)) == [0.50, 0.75]
        finally:
            sys.path.pop(0)


def test_paper_run_one_direct_signal_does_not_compile_full_output(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_project(root, package_name="direct_signal", engine_code=DIRECT_SIGNAL_ENGINE_CODE, days=4)
        sys.path.insert(0, str(root))

        try:
            write_bootstrap_log(
                "data/trade_log_direct_signal.csv",
                dates=["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
                closes=[100.0, 101.0],
                positions=[0.0, 0.0],
            )
            cfg = load_config()

            result = paper_run_one(
                cfg["strategies"][0],
                settings=cfg.get("settings"),
                as_of="2026-01-04T00:00:00Z",
            )

            engine_module = importlib.import_module("strategies.direct_signal.engine")
            assert engine_module.DirectSignalEngine.calls == [
                ("2026-01-02", 2),
                ("2026-01-03", 3),
                ("2026-01-04", 4),
            ]
            assert result["execution_mode"] == "direct_paper_signal"
            assert result["n_rows"] == 2
            paper_df = read_trade_log("data/paper_log_direct_signal.csv")
            assert list(paper_df["date"].dt.strftime("%Y-%m-%d")) == [
                "2026-01-03",
                "2026-01-04",
            ]
        finally:
            sys.path.pop(0)


def test_paper_history_fixed_lookback_limits_compiled_recompute(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_project(
            root,
            package_name="decision_profile",
            engine_code=DECISION_ENGINE_CODE,
            profile_yaml="""
      history:
        boundary: fixed_lookback
        lookbackBars: 1
        feeds:
          - ETHUSD
""".rstrip(),
        )
        sys.path.insert(0, str(root))

        try:
            write_bootstrap_log(
                "data/trade_log_decision_profile.csv",
                dates=["2026-01-23T00:00:00Z"],
                closes=[122.0],
                positions=[0.18],
                next_positions=[0.18],
            )
            cfg = load_config()
            cfg["strategies"][0]["runtime"] = {
                "paperExecutionProfile": cfg["strategies"][0].pop("paper_execution_profile")
            }

            result = paper_run_one(
                cfg["strategies"][0],
                settings=cfg.get("settings"),
                as_of="2026-01-25T00:00:00Z",
            )

            engine_module = importlib.import_module("strategies.decision_profile.engine")
            assert engine_module.CountingDecisionEngine.calls == [
                (
                    "2026-01-05",
                    "2026-01-25",
                    SYSTEM_LOOKBACK_PADDING_BARS + 1,
                )
            ]
            assert result["execution_mode"] == "compiled_output"
            assert result["paper_history_boundary"]["boundary"] == "fixed_lookback"
            paper_df = read_trade_log("data/paper_log_decision_profile.csv")
            assert list(paper_df["date"].dt.strftime("%Y-%m-%d")) == [
                "2026-01-24",
                "2026-01-25",
            ]
            assert list(paper_df["next_position"].round(2)) == [4.75, 5.00]
        finally:
            sys.path.pop(0)


def test_paper_history_fixed_lookback_expands_compiled_recompute_to_cover_cursor(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_project(
            root,
            package_name="decision_profile_catchup",
            engine_code=DECISION_ENGINE_CODE,
            profile_yaml="""
      history:
        boundary: fixed_lookback
        lookbackBars: 1
        feeds:
          - ETHUSD
""".rstrip(),
        )
        sys.path.insert(0, str(root))

        try:
            write_bootstrap_log(
                "data/trade_log_decision_profile_catchup.csv",
                dates=["2026-01-02T00:00:00Z"],
                closes=[101.0],
                positions=[0.18],
                next_positions=[0.18],
            )
            cfg = load_config()

            result = paper_run_one(
                cfg["strategies"][0],
                settings=cfg.get("settings"),
                as_of="2026-01-25T00:00:00Z",
            )

            engine_module = importlib.import_module("strategies.decision_profile_catchup.engine")
            assert engine_module.CountingDecisionEngine.calls == [
                ("2026-01-01", "2026-01-25", 25)
            ]
            assert result["execution_mode"] == "compiled_output"
            assert result["paper_history_boundary"]["boundary"] == "fixed_lookback"
            assert result["n_rows"] == 23
            paper_df = read_trade_log("data/paper_log_decision_profile_catchup.csv")
            assert list(paper_df["date"].dt.strftime("%Y-%m-%d")) == [
                f"2026-01-{day:02d}" for day in range(3, 26)
            ]
            assert float(paper_df["next_position"].iloc[0]) == 0.50
            assert float(paper_df["next_position"].iloc[-1]) == 6.00
        finally:
            sys.path.pop(0)


def test_paper_history_origin_anchored_limits_compiled_recompute(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_project(
            root,
            package_name="decision_origin",
            engine_code=DECISION_ENGINE_CODE,
            profile_yaml="""
      history:
        boundary: origin_anchored
        origin: "2026-01-10T00:00:00Z"
        feeds:
          - ETHUSD
""".rstrip(),
        )
        sys.path.insert(0, str(root))

        try:
            write_bootstrap_log(
                "data/trade_log_decision_origin.csv",
                dates=["2026-01-23T00:00:00Z"],
                closes=[122.0],
                positions=[3.25],
                next_positions=[3.25],
            )
            cfg = load_config()

            result = paper_run_one(
                cfg["strategies"][0],
                settings=cfg.get("settings"),
                as_of="2026-01-25T00:00:00Z",
            )

            engine_module = importlib.import_module("strategies.decision_origin.engine")
            assert engine_module.CountingDecisionEngine.calls == [
                ("2026-01-10", "2026-01-25", 16)
            ]
            assert result["execution_mode"] == "compiled_output"
            assert result["paper_history_boundary"]["boundary"] == "origin_anchored"
            paper_df = read_trade_log("data/paper_log_decision_origin.csv")
            assert list(paper_df["date"].dt.strftime("%Y-%m-%d")) == [
                "2026-01-24",
                "2026-01-25",
            ]
            assert list(paper_df["next_position"].round(2)) == [3.50, 3.75]
        finally:
            sys.path.pop(0)


def test_paper_history_fixed_lookback_limits_direct_signal_reads(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_project(
            root,
            package_name="direct_profile",
            engine_code=DIRECT_SIGNAL_ENGINE_CODE,
            profile_yaml="""
      history:
        boundary: fixed_lookback
        lookbackBars: 1
        feeds:
          - ETHUSD
""".rstrip(),
        )
        sys.path.insert(0, str(root))

        try:
            write_bootstrap_log(
                "data/trade_log_direct_profile.csv",
                dates=["2026-01-23T00:00:00Z"],
                closes=[122.0],
                positions=[1.0],
                next_positions=[1.0],
            )
            cfg = load_config()

            result = paper_run_one(
                cfg["strategies"][0],
                settings=cfg.get("settings"),
                as_of="2026-01-25T00:00:00Z",
            )

            engine_module = importlib.import_module("strategies.direct_profile.engine")
            assert engine_module.DirectSignalEngine.calls == [
                ("2026-01-24", SYSTEM_LOOKBACK_PADDING_BARS + 1),
                ("2026-01-25", SYSTEM_LOOKBACK_PADDING_BARS + 1),
            ]
            assert result["execution_mode"] == "direct_paper_signal"
            assert result["paper_history_boundary"]["boundary"] == "fixed_lookback"
            assert result["n_rows"] == 2
        finally:
            sys.path.pop(0)


def test_paper_bootstrap_context_bypasses_daily_history_window(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        write_project(
            root,
            package_name="bootstrap_context",
            engine_code=BOOTSTRAP_CONTEXT_ENGINE_CODE,
            profile_yaml="""
      history:
        boundary: fixed_lookback
        lookbackBars: 1
        feeds:
          - ETHUSD
""".rstrip(),
        )
        sys.path.insert(0, str(root))

        try:
            write_bootstrap_log(
                "data/trade_log_bootstrap_context.csv",
                dates=["2026-01-23T00:00:00Z"],
                closes=[122.0],
                positions=[1.0],
                next_positions=[1.0],
            )
            cfg = load_config()
            strategy = cfg["strategies"][0]
            strategy["_paper_data_window"] = {
                "boundary": "fixed_lookback",
                "limit": SYSTEM_LOOKBACK_PADDING_BARS + 1,
                "source": "paper_execution_profile",
            }
            engine_module = importlib.import_module("strategies.bootstrap_context.engine")
            engine = engine_module.BootstrapContextEngine(context=strategy)

            with engine.paper_bootstrap_cutover_scope("2026-01-20T00:00:00Z"):
                engine.build_paper_initial_state(cutover_as_of="2026-01-20T00:00:00Z")

                explicit_ctx = engine.paper_bootstrap_context(
                    start="2026-01-01T00:00:00Z",
                    end="2026-01-20T00:00:00Z",
                )
                explicit_close = explicit_ctx.target.series("close")
                assert str(explicit_close.index[-1].date()) == "2026-01-20"

                daily_ctx = engine.decision_context(start="2026-01-01T00:00:00Z")
                daily_close = daily_ctx.target.series("close")
                assert str(daily_close.index[-1].date()) == "2026-01-20"

                with pytest.raises(
                    ValueError,
                    match=(
                        "paper bootstrap context end 2026-01-21 is after "
                        "validation cutover 2026-01-20"
                    ),
                ):
                    engine.paper_bootstrap_context(
                        start="2026-01-01T00:00:00Z",
                        end="2026-01-21T00:00:00Z",
                    )

                with pytest.raises(
                    ValueError,
                    match=(
                        "paper bootstrap context end 2026-01-21 is after "
                        "validation cutover 2026-01-20"
                    ),
                ):
                    engine.decision_context(
                        start="2026-01-01T00:00:00Z",
                        end="2026-01-21T00:00:00Z",
                    )

            engine.get_paper_signal(as_of="2026-01-25T00:00:00Z")

            assert engine_module.BootstrapContextEngine.calls == [
                ("bootstrap", "2026-01-01", "2026-01-20", 20),
                ("daily", "2026-01-05", "2026-01-25", SYSTEM_LOOKBACK_PADDING_BARS + 1),
            ]
        finally:
            sys.path.pop(0)
