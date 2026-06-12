from pathlib import Path
import importlib
import sys

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from abel_edge.config import load_config
from abel_edge.cli import main
from abel_edge.engine.ledger import read_trade_log, write_trade_log
from abel_edge.engine.trader import paper_run_one


ENGINE_CODE = """
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, datetime, timezone

from abel_edge.engine.base import StrategyEngine


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
        return {
            'next_position': 1.0 if last_close >= 100.0 else 0.0,
            'date': str(target['timestamp'].iloc[-1]),
            'price': last_close,
            'data_backend': 'csv',
            'data_fetch_status': 'provider_fetch',
            'data_latest_timestamp': str(target['timestamp'].iloc[-1]),
            'broker_decision_time': datetime(2026, 1, 4, 16, 30, tzinfo=timezone.utc),
            'provider_session_date': date(2026, 1, 4),
            'provider_sequence': np.int64(len(target)),
            'strategy_label': self.context.get('id'),
            'paper_audit_status': 'provider_fetch',
            'source': 'bad-source-override',
            'pnl': 999.0,
            'cum_return': 999.0,
            'gross_pnl': 999.0,
            'turnover': 999.0,
            'execution_cost': 999.0,
        }
""".strip()


def _write_strategy_project(tmp_path: Path) -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
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
    (tmp_path / "data" / "btcusd.csv").write_text(
        "timestamp,close\n"
        "2026-01-01T00:00:00Z,50\n"
        "2026-01-02T00:00:00Z,70\n"
        "2026-01-03T00:00:00Z,130\n"
        "2026-01-04T00:00:00Z,80\n",
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
    paper_log: data/paper_log_paper_demo.csv
    price_data:
      source: csv
      path: data/ethusd.csv
  - id: paper_alt
    name: "Paper Alt"
    asset: BTCUSD
    color: "#F59E0B"
    engine: strategies.paper_demo.engine
    trade_log: data/trade_log_paper_alt.csv
    paper_log: data/paper_log_paper_alt.csv
    price_data:
      source: csv
      path: data/btcusd.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_bootstrap_log(path: str, *, closes: list[float], positions: list[float]) -> None:
    dates = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"], utc=True)
    returns = [0.0, closes[1] / closes[0] - 1.0]
    pnl = [position * value for position, value in zip(positions, returns)]
    write_trade_log(
        dates,
        returns,
        pnl,
        positions,
        path,
        close_prices=closes,
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
        trade_df = read_trade_log("data/trade_log_paper_demo.csv")
        paper_df = read_trade_log("data/paper_log_paper_demo.csv")
        assert len(trade_df) == 2
        assert list(paper_df["source"].tail(2)) == ["live", "live"]
        assert list(paper_df["date"].dt.strftime("%Y-%m-%d").tail(2)) == [
            "2026-01-03",
            "2026-01-04",
        ]
        assert list(paper_df["position"].tail(2).round(2)) == [1.0, 0.0]
        assert list(paper_df["next_position"].tail(2).round(2)) == [0.0, 1.0]
        assert list(paper_df["close"].tail(2).round(2)) == [90.0, 120.0]
        assert float(paper_df.iloc[-2]["pnl"]) < 0
        assert paper_df.iloc[-1]["data_backend"] == "csv"
        assert paper_df.iloc[-1]["data_fetch_status"] == "provider_fetch"
        assert str(paper_df.iloc[-1]["data_latest_timestamp"]).startswith("2026-01-04")
        assert str(paper_df.iloc[-1]["broker_decision_time"]).startswith("2026-01-04T16:30:00")
        assert str(paper_df.iloc[-1]["provider_session_date"]) == "2026-01-04"
        assert int(paper_df.iloc[-1]["provider_sequence"]) == 4
        assert paper_df.iloc[-1]["paper_audit_status"] == "provider_fetch"
        assert paper_df.iloc[-1]["strategy_label"] == "paper_demo"
        assert paper_df.iloc[-1]["source"] == "live"
        assert float(paper_df.iloc[-1]["pnl"]) == 0.0
        assert float(paper_df.iloc[-1]["cum_return"]) != 999.0
        for column in ["gross_pnl", "turnover", "execution_cost"]:
            if column in paper_df.columns:
                assert float(paper_df.iloc[-1][column]) != 999.0
        sys.path.pop(0)


def test_paper_run_one_matches_golden_append_math(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_strategy_project(root)
        sys.path.insert(0, str(root))

        try:
            _write_bootstrap_log(
                "data/trade_log_paper_demo.csv",
                closes=[100.0, 110.0],
                positions=[0.0, 1.0],
            )
            cfg = load_config()
            strategy = next(item for item in cfg["strategies"] if item["id"] == "paper_demo")

            result = paper_run_one(
                strategy,
                settings=cfg.get("settings"),
                as_of="2026-01-04T00:00:00Z",
            )

            paper_df = read_trade_log("data/paper_log_paper_demo.csv")
            live = paper_df[paper_df["source"] == "live"].reset_index(drop=True)
            assert result["n_rows"] == 2
            assert list(live["date"].dt.strftime("%Y-%m-%d")) == ["2026-01-03", "2026-01-04"]
            assert list(live["close"]) == pytest.approx([90.0, 120.0])
            assert list(live["position"]) == pytest.approx([1.0, 0.0])
            assert list(live["next_position"]) == pytest.approx([0.0, 1.0])
            assert list(live["asset_return"]) == pytest.approx([
                90.0 / 110.0 - 1.0,
                120.0 / 90.0 - 1.0,
            ])
            assert list(live["pnl"]) == pytest.approx([
                1.0 * (90.0 / 110.0 - 1.0),
                0.0 * (120.0 / 90.0 - 1.0),
            ])
            assert list(live["cum_return"]) == pytest.approx(
                np.cumprod(1.0 + live["pnl"].to_numpy(dtype=float)) - 1.0
            )
            assert result["latest_snapshot"]["last_processed_date"] == "2026-01-04T00:00:00+00:00"
            assert result["latest_snapshot"]["current_position"] == 0.0
            assert result["latest_snapshot"]["next_position"] == 1.0
            assert result["latest_snapshot"]["latest_close"] == 120.0
        finally:
            sys.path.pop(0)


def test_paper_run_one_returns_latest_snapshot_for_each_strategy(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_strategy_project(root)
        sys.path.insert(0, str(root))

        try:
            _write_bootstrap_log(
                "data/trade_log_paper_demo.csv",
                closes=[100.0, 110.0],
                positions=[0.0, 1.0],
            )
            _write_bootstrap_log(
                "data/trade_log_paper_alt.csv",
                closes=[50.0, 70.0],
                positions=[0.0, 0.0],
            )
            cfg = load_config()
            strategies = {item["id"]: item for item in cfg["strategies"]}

            demo = paper_run_one(
                strategies["paper_demo"],
                settings=cfg.get("settings"),
                as_of="2026-01-04T00:00:00Z",
            )
            alt = paper_run_one(
                strategies["paper_alt"],
                settings=cfg.get("settings"),
                as_of="2026-01-04T00:00:00Z",
            )

            expected_demo = {
                "strategy_id": "paper_demo",
                "asset": "ETHUSD",
                "last_processed_date": "2026-01-04T00:00:00+00:00",
                "current_position": 0.0,
                "next_position": 1.0,
                "latest_close": 120.0,
                "source": "live",
                "strategy_label": "paper_demo",
                "data_backend": "csv",
                "data_fetch_status": "provider_fetch",
                "data_latest_timestamp": "2026-01-04 00:00:00+00:00",
                "paper_audit_status": "provider_fetch",
            }
            for key, value in expected_demo.items():
                assert demo["latest_snapshot"][key] == value
            assert int(demo["latest_snapshot"]["provider_sequence"]) == 4

            expected_alt = {
                "strategy_id": "paper_alt",
                "asset": "BTCUSD",
                "last_processed_date": "2026-01-04T00:00:00+00:00",
                "current_position": 1.0,
                "next_position": 0.0,
                "latest_close": 80.0,
                "source": "live",
                "strategy_label": "paper_alt",
                "data_backend": "csv",
                "data_fetch_status": "provider_fetch",
                "data_latest_timestamp": "2026-01-04 00:00:00+00:00",
                "paper_audit_status": "provider_fetch",
            }
            for key, value in expected_alt.items():
                assert alt["latest_snapshot"][key] == value
            assert int(alt["latest_snapshot"]["provider_sequence"]) == 4
        finally:
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

        trade_df = read_trade_log("data/trade_log_paper_demo.csv")
        paper_df = read_trade_log("data/paper_log_paper_demo.csv")
        assert len(trade_df) == 2
        assert len(paper_df) == 2
        assert int((paper_df["source"] == "live").sum()) == 2
        sys.path.pop(0)
