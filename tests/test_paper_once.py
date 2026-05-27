from pathlib import Path
import importlib
import sys

import pandas as pd
from click.testing import CliRunner

from abel_edge.config import load_config
from abel_edge.engine.ledger import read_trade_log, write_trade_log
from abel_edge.engine.trader import SYSTEM_LOOKBACK_PADDING_BARS, paper_run_one


DECISION_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class CountingDecisionEngine(StrategyEngine):
    calls = []

    def compute_decisions(self, ctx):
        close = ctx.target.series("close")
        type(self).calls.append((str(close.index[0].date()), str(close.index[-1].date()), len(close)))
        next_position = pd.Series(
            [idx * 0.25 for idx in range(len(close))],
            index=close.index,
        )
        return ctx.decisions(next_position)
""".strip()


DIRECT_SIGNAL_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class DirectSignalEngine(StrategyEngine):
    calls = []

    def compute_decisions(self, ctx):
        raise AssertionError("paper_run_one direct mode must not compile full output")

    def get_paper_signal(self, *, as_of=None):
        bars = self.load_bars(end=as_of).sort_values("timestamp")
        last = bars.iloc[-1]
        type(self).calls.append((str(pd.to_datetime(last["timestamp"], utc=True).date()), len(bars)))
        return {
            "next_position": 1.0 if float(last["close"]) >= 120.0 else 0.0,
            "data_latest_timestamp": str(pd.to_datetime(last["timestamp"], utc=True)),
        }
""".strip()


def _clear_strategy_modules() -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _daily_price_csv(*, days: int = 25) -> str:
    rows = ["timestamp,close"]
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for idx in range(days):
        ts = start + pd.Timedelta(days=idx)
        rows.append(f"{ts.isoformat()},{100 + idx}")
    return "\n".join(rows) + "\n"


def _write_project(
    root: Path,
    *,
    package_name: str,
    engine_code: str,
    profile_yaml: str | None = None,
    days: int = 25,
) -> None:
    _clear_strategy_modules()
    strategy_dir = root / "strategies" / package_name
    strategy_dir.mkdir(parents=True)
    (root / "strategies" / "__init__.py").write_text("", encoding="utf-8")
    (strategy_dir / "__init__.py").write_text("", encoding="utf-8")
    (strategy_dir / "engine.py").write_text(engine_code, encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "ethusd.csv").write_text(_daily_price_csv(days=days), encoding="utf-8")
    profile_block = f"\n    paper_execution_profile:\n{profile_yaml}" if profile_yaml else ""
    (root / "strategies.yaml").write_text(
        f"""
settings:
  price_data:
    default_source: csv
    default_timeframe: 1d
strategies:
  - id: {package_name}
    name: "{package_name}"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.{package_name}.engine
    trade_log: data/trade_log_{package_name}.csv
    paper_log: data/paper_log_{package_name}.csv
    price_data:
      source: csv
      path: data/ethusd.csv{profile_block}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_bootstrap_log(
    path: str,
    *,
    dates: list[str],
    closes: list[float],
    positions: list[float],
    next_positions: list[float] | None = None,
) -> None:
    parsed = pd.to_datetime(dates, utc=True)
    returns = [0.0]
    returns.extend(closes[idx] / closes[idx - 1] - 1.0 for idx in range(1, len(closes)))
    pnl = [position * value for position, value in zip(positions, returns)]
    write_trade_log(
        parsed,
        returns,
        pnl,
        positions,
        path,
        close_prices=closes,
        next_positions=next_positions,
    )


def test_paper_run_one_default_signal_computes_once_and_rolls_position(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_project(root, package_name="decision_once", engine_code=DECISION_ENGINE_CODE, days=4)
        sys.path.insert(0, str(root))

        try:
            _write_bootstrap_log(
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
        _write_project(root, package_name="direct_signal", engine_code=DIRECT_SIGNAL_ENGINE_CODE, days=4)
        sys.path.insert(0, str(root))

        try:
            _write_bootstrap_log(
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
        _write_project(
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
            _write_bootstrap_log(
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


def test_paper_history_origin_anchored_limits_compiled_recompute(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_project(
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
            _write_bootstrap_log(
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
        _write_project(
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
            _write_bootstrap_log(
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
