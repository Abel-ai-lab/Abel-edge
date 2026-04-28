"""Regression tests for execution settings on the run path."""

import importlib
import sys
from pathlib import Path

from click.testing import CliRunner

from abel_edge.cli import main
from abel_edge.engine.ledger import read_trade_log

ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class ExecutionDemoEngine(StrategyEngine):
    def compute_signals(self):
        bars = self.load_bars(limit=3)
        target = bars[bars['symbol'] == self.context.get('asset', 'ETHUSD')].copy().sort_values('timestamp')
        prices = target['close'].astype(float).to_numpy()
        dates = pd.DatetimeIndex(target['timestamp'])
        positions = pd.Series([0.0, 2.0, -2.0], dtype=float).to_numpy()
        return positions, dates, prices

    def get_latest_signal(self):
        return {'position': 0.0}
""".strip()

DECISION_ENGINE_CODE = """
from __future__ import annotations

from abel_edge.engine.base import StrategyEngine


class ExecutionDemoEngine(StrategyEngine):
    def compute_decisions(self, ctx):
        close = ctx.target.series('close')
        next_position = (close.pct_change().fillna(0.0) > 0).astype(float)
        if len(next_position) > 0:
            next_position.iloc[0] = 0.0
        return ctx.decisions(next_position)
""".strip()


def _write_strategy_project(
    tmp_path: Path,
    *,
    execution_block: str = "",
    engine_code: str = ENGINE_CODE,
) -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "strategies" / "execution_demo").mkdir()
    (tmp_path / "strategies" / "execution_demo" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "strategies" / "execution_demo" / "engine.py").write_text(
        engine_code, encoding="utf-8"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ethusd.csv").write_text(
        "timestamp,close\n"
        "2026-01-01T00:00:00Z,100\n"
        "2026-01-02T00:00:00Z,110\n"
        "2026-01-03T00:00:00Z,99\n",
        encoding="utf-8",
    )
    (tmp_path / "strategies.yaml").write_text(
        f"""
settings:
  price_data:
    default_adapter: csv
    default_timeframe: 1d
{execution_block}strategies:
  - id: execution_demo
    name: "Execution Demo"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.execution_demo.engine
    trade_log: data/trade_log_execution_demo.csv
    price_data:
      adapter: csv
      path: data/ethusd.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_run_writes_execution_columns_and_clipped_positions(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_strategy_project(
            root,
            execution_block="  execution:\n    cost_bps: 100\n    max_abs_position: 0.5\n",
        )
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "execution_demo"])

        assert result.exit_code == 0, result.output
        trade_df = read_trade_log("data/trade_log_execution_demo.csv")
        assert list(trade_df["position"].round(2)) == [0.0, 0.5, -0.5]
        assert "decision_time" in trade_df.columns
        assert "effective_time" in trade_df.columns
        assert list(trade_df["turnover"].round(2)) == [0.0, 0.5, 1.0]
        assert list(trade_df["execution_cost"].round(3)) == [0.0, 0.005, 0.01]
        assert list(trade_df["gross_pnl"].round(3)) == [0.0, 0.05, 0.05]
        assert list(trade_df["pnl"].round(3)) == [0.0, 0.045, 0.04]
        sys.path.pop(0)


def test_run_defaults_to_legacy_execution_when_settings_missing(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_strategy_project(root)
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "execution_demo"])

        assert result.exit_code == 0, result.output
        trade_df = read_trade_log("data/trade_log_execution_demo.csv")
        assert list(trade_df["position"].round(2)) == [0.0, 2.0, -2.0]
        assert list(trade_df["execution_cost"].round(3)) == [0.0, 0.0, 0.0]
        assert list(trade_df["pnl"].round(2)) == [0.0, 0.2, 0.2]
        sys.path.pop(0)


def test_run_supports_decision_context_engines(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_strategy_project(root, engine_code=DECISION_ENGINE_CODE)
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "execution_demo"])

        assert result.exit_code == 0, result.output
        trade_df = read_trade_log("data/trade_log_execution_demo.csv")
        assert list(trade_df["next_position"].round(2)) == [0.0, 1.0, 0.0]
        assert list(trade_df["position"].round(2)) == [0.0, 0.0, 1.0]
        assert "decision_time" in trade_df.columns
        assert "effective_time" in trade_df.columns
        sys.path.pop(0)
