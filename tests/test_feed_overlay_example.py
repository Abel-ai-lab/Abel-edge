"""Regression test for the bundled framework-managed feed example."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.engine.ledger import read_trade_log


def _reset_strategy_modules() -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def test_feed_overlay_example_runs_with_declared_feed_path(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        strategies_dir = root / "strategies"
        strategies_dir.mkdir()
        (strategies_dir / "__init__.py").write_text("", encoding="utf-8")

        example_dir = Path(__file__).resolve().parent.parent / "examples" / "feed_overlay_demo"
        shutil.copytree(example_dir, strategies_dir / "feed_overlay_demo")

        data_dir = root / "data"
        data_dir.mkdir()
        (data_dir / "ethusd.csv").write_text(
            "timestamp,close\n"
            "2026-01-01,100\n"
            "2026-01-02,102\n"
            "2026-01-03,104\n"
            "2026-01-04,106\n",
            encoding="utf-8",
        )
        (data_dir / "btcusd.csv").write_text(
            "timestamp,close\n"
            "2026-01-01,20\n"
            "2026-01-02,30\n"
            "2026-01-03,40\n"
            "2026-01-04,50\n",
            encoding="utf-8",
        )
        (data_dir / "risk_scale.csv").write_text(
            "timestamp,value\n"
            "2026-01-01,0.2\n"
            "2026-01-02,0.4\n"
            "2026-01-03,0.6\n"
            "2026-01-04,0.8\n",
            encoding="utf-8",
        )
        (root / "strategies.yaml").write_text(
            """
settings:
  price_data:
    default_source: csv
    default_timeframe: 1d
strategies:
  - id: feed_overlay_demo
    name: "Feed Overlay Demo"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.feed_overlay_demo.engine
    trade_log: data/trade_log_feed_overlay_demo.csv
    price_data:
      source: csv
      path: data/ethusd.csv
    feeds:
      btc_ref:
        kind: bars
        source: csv
        path: data/btcusd.csv
        symbol: BTCUSD
      risk_scale:
        kind: series
        source: csv
        path: data/risk_scale.csv
        field: value
""".strip()
            + "\n",
            encoding="utf-8",
        )

        _reset_strategy_modules()
        sys.path.insert(0, str(root))
        try:
            result = runner.invoke(main, ["run", "--strategy", "feed_overlay_demo"])
            assert result.exit_code == 0, result.output
            trade_df = read_trade_log("data/trade_log_feed_overlay_demo.csv")
            assert list(trade_df["position"].round(2)) == [0.0, 0.0, 0.6, 0.8]
        finally:
            sys.path.pop(0)
            _reset_strategy_modules()
