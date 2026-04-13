from pathlib import Path
import sys

import pandas as pd
from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.engine.ledger import read_trade_log, write_trade_log


ENGINE_CODE = """
from __future__ import annotations

import numpy as np
import pandas as pd

from causal_edge.engine.base import StrategyEngine


class PaperDemoEngine(StrategyEngine):
    def compute_signals(self):
        bars = self.load_bars(limit=10)
        target = bars[bars['symbol'] == self.context.get('asset', 'ETHUSD')].copy().sort_values('timestamp')
        prices = target['close'].astype(float).to_numpy()
        dates = pd.DatetimeIndex(target['timestamp'])
        prev_close = pd.Series(prices).shift(1)
        positions = (prev_close >= 100.0).astype(float).fillna(0.0).to_numpy()
        return positions, dates, prices

    def get_latest_signal(self):
        positions, dates, prices = self.compute_signals()
        return {'position': float(positions[-1]), 'date': str(dates[-1]), 'price': float(prices[-1])}

    def get_paper_signal(self, *, as_of=None):
        bars = self.load_bars(limit=10, end=as_of)
        target = bars[bars['symbol'] == self.context.get('asset', 'ETHUSD')].copy().sort_values('timestamp')
        last_close = float(target['close'].iloc[-1])
        return {'next_position': 1.0 if last_close >= 100.0 else 0.0, 'date': str(target['timestamp'].iloc[-1]), 'price': last_close}
""".strip()


def _write_strategy_project(tmp_path: Path) -> None:
    (tmp_path / "strategies").mkdir()
    (tmp_path / "strategies" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "strategies" / "paper_demo").mkdir()
    (tmp_path / "strategies" / "paper_demo" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "strategies" / "paper_demo" / "engine.py").write_text(
        ENGINE_CODE, encoding="utf-8"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ethusd.csv").write_text(
        "timestamp,close\n"
        "2026-01-01T00:00:00Z,100\n"
        "2026-01-02T00:00:00Z,110\n"
        "2026-01-03T00:00:00Z,90\n"
        "2026-01-04T00:00:00Z,120\n",
        encoding="utf-8",
    )
    (tmp_path / "strategies.yaml").write_text(
        """
settings:
  price_data:
    default_source: csv
    default_timeframe: 1d
strategies:
  - id: paper_demo
    name: "Paper Demo"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.paper_demo.engine
    trade_log: data/trade_log_paper_demo.csv
    price_data:
      source: csv
      path: data/ethusd.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_paper_appends_live_rows_with_close_fill_semantics(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_strategy_project(root)
        sys.path.insert(0, str(root))

        dates = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True)
        returns = [0.0, 0.10]
        pnl = [0.0, 0.10]
        positions = [0.0, 1.0]
        write_trade_log(
            dates,
            returns,
            pnl,
            positions,
            "data/trade_log_paper_demo.csv",
            close_prices=[100.0, 110.0],
        )

        result = runner.invoke(
            main,
            ["paper", "--strategy", "paper_demo", "--as-of", "2026-01-04T00:00:00Z"],
        )

        assert result.exit_code == 0, result.output
        df = read_trade_log("data/trade_log_paper_demo.csv")
        assert list(df["source"].tail(2)) == ["live", "live"]
        assert list(df["date"].dt.strftime("%Y-%m-%d").tail(2)) == ["2026-01-03", "2026-01-04"]
        assert list(df["position"].tail(2).round(2)) == [1.0, 0.0]
        assert list(df["next_position"].tail(2).round(2)) == [0.0, 1.0]
        assert list(df["close"].tail(2).round(2)) == [90.0, 120.0]
        assert float(df.iloc[-2]["pnl"]) < 0
        assert float(df.iloc[-1]["pnl"]) == 0.0
        sys.path.pop(0)


def test_paper_is_idempotent_when_no_new_bars(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_strategy_project(root)
        sys.path.insert(0, str(root))

        dates = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True)
        returns = [0.0, 0.10]
        pnl = [0.0, 0.10]
        positions = [0.0, 1.0]
        write_trade_log(
            dates,
            returns,
            pnl,
            positions,
            "data/trade_log_paper_demo.csv",
            close_prices=[100.0, 110.0],
        )

        first = runner.invoke(
            main,
            ["paper", "--strategy", "paper_demo", "--as-of", "2026-01-04T00:00:00Z"],
        )
        assert first.exit_code == 0, first.output

        second = runner.invoke(
            main,
            ["paper", "--strategy", "paper_demo", "--as-of", "2026-01-04T00:00:00Z"],
        )
        assert second.exit_code == 0, second.output
        assert "no new closed bars" in second.output

        df = read_trade_log("data/trade_log_paper_demo.csv")
        assert len(df) == 4
        assert int((df["source"] == "live").sum()) == 2
        sys.path.pop(0)
