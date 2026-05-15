from pathlib import Path
import importlib
import sys

import pandas as pd
from click.testing import CliRunner

from abel_edge.config import load_config
from abel_edge.engine.ledger import read_trade_log, write_trade_log
from abel_edge.engine.trader import paper_run_one


DECISION_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class CountingDecisionEngine(StrategyEngine):
    calls = 0

    def compute_decisions(self, ctx):
        type(self).calls += 1
        close = ctx.target.series("close")
        next_position = pd.Series(
            [idx * 0.25 for idx in range(len(close))],
            index=close.index,
        )
        return ctx.decisions(next_position)
""".strip()


def _clear_strategy_modules() -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _write_decision_project(root: Path) -> None:
    _clear_strategy_modules()
    strategy_dir = root / "strategies" / "decision_once"
    strategy_dir.mkdir(parents=True)
    (root / "strategies" / "__init__.py").write_text("", encoding="utf-8")
    (strategy_dir / "__init__.py").write_text("", encoding="utf-8")
    (strategy_dir / "engine.py").write_text(DECISION_ENGINE_CODE, encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "ethusd.csv").write_text(
        "timestamp,close\n"
        "2026-01-01T00:00:00Z,100\n"
        "2026-01-02T00:00:00Z,110\n"
        "2026-01-03T00:00:00Z,120\n"
        "2026-01-04T00:00:00Z,130\n",
        encoding="utf-8",
    )
    (root / "strategies.yaml").write_text(
        """
settings:
  price_data:
    default_source: csv
    default_timeframe: 1d
strategies:
  - id: decision_once
    name: "Decision Once"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.decision_once.engine
    trade_log: data/trade_log_decision_once.csv
    paper_log: data/paper_log_decision_once.csv
    price_data:
      source: csv
      path: data/ethusd.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_paper_run_one_default_signal_computes_once_and_rolls_position(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_decision_project(root)
        sys.path.insert(0, str(root))

        try:
            dates = pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
                utc=True,
            )
            write_trade_log(
                dates,
                [0.0, 0.10],
                [0.0, 0.0],
                [0.0, 0.0],
                "data/trade_log_decision_once.csv",
                close_prices=[100.0, 110.0],
                next_positions=[0.0, 0.25],
            )
            cfg = load_config()

            result = paper_run_one(
                cfg["strategies"][0],
                settings=cfg.get("settings"),
                as_of="2026-01-04T00:00:00Z",
            )

            engine_module = importlib.import_module("strategies.decision_once.engine")
            assert engine_module.CountingDecisionEngine.calls == 1
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
