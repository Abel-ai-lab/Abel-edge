"""Runtime contract tests for feed loading, alignment, and signal outputs."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from causal_edge.cli import main
from causal_edge.config import load_config
from causal_edge.engine.feed_contract import FeedAlignmentError, align_series_to_dates
from causal_edge.engine.ledger import read_trade_log
from causal_edge.engine.signal_contract import SignalContractError, validate_signal_output


FEED_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from causal_edge.engine.base import StrategyEngine


class FeedDemoEngine(StrategyEngine):
    def compute_signals(self):
        bars = self.load_bars(limit=3)
        target = bars[bars['symbol'] == self.context.get('asset', 'ETHUSD')].copy().sort_values('timestamp')
        dates = pd.DatetimeIndex(target['timestamp'])
        prices = target['close'].astype(float).to_numpy()
        scale = self.feed_series('risk_scale', align_to=dates, method='ffill', allow_gaps=False)
        positions = scale.astype(float).to_numpy()
        return self.finalize_signals(positions, dates, prices)

    def get_latest_signal(self):
        return {'position': 0.0}
""".strip()


BAR_FEED_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from causal_edge.engine.base import StrategyEngine


class BarsFeedDemoEngine(StrategyEngine):
    def compute_signals(self):
        bars = self.load_bars(limit=3)
        target = bars[bars['symbol'] == self.context.get('asset', 'ETHUSD')].copy().sort_values('timestamp')
        dates = pd.DatetimeIndex(target['timestamp'])
        prices = target['close'].astype(float).to_numpy()
        ref_close = self.feed_series('btc_ref', field='close', align_to=dates, method='ffill', allow_gaps=False)
        positions = (ref_close.astype(float) / 100.0).to_numpy()
        return self.finalize_signals(positions, dates, prices)

    def get_latest_signal(self):
        return {'position': 0.0}
""".strip()


UNDECLARED_FEED_ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from causal_edge.engine.base import StrategyEngine


class UndeclaredFeedEngine(StrategyEngine):
    def compute_signals(self):
        bars = self.load_bars(limit=3)
        target = bars[bars['symbol'] == self.context.get('asset', 'ETHUSD')].copy().sort_values('timestamp')
        dates = pd.DatetimeIndex(target['timestamp'])
        prices = target['close'].astype(float).to_numpy()
        scale = self.feed_series('missing_feed', align_to=dates, method='ffill', allow_gaps=False)
        positions = scale.astype(float).to_numpy()
        return self.finalize_signals(positions, dates, prices)

    def get_latest_signal(self):
        return {'position': 0.0}
""".strip()


NAIVE_ENGINE_CODE = """
from __future__ import annotations

import numpy as np
import pandas as pd

from causal_edge.engine.base import StrategyEngine


class NaiveDatesEngine(StrategyEngine):
    def compute_signals(self):
        return (
            np.array([0.0, 1.0, 0.0], dtype=float),
            pd.date_range('2026-01-01', periods=3),
            np.array([100.0, 110.0, 120.0], dtype=float),
        )

    def get_latest_signal(self):
        return {'position': 0.0}
""".strip()


PRIMARY_ONLY_ENGINE_CODE = """
from __future__ import annotations

import numpy as np
import pandas as pd

from causal_edge.engine.base import StrategyEngine


class PrimaryOnlyEngine(StrategyEngine):
    def compute_signals(self):
        bars = self.load_bars(limit=3)
        target = bars[bars['symbol'] == self.context.get('asset', 'ETHUSD')].copy().sort_values('timestamp')
        dates = pd.DatetimeIndex(target['timestamp'])
        prices = target['close'].astype(float).to_numpy()
        positions = np.zeros(len(target), dtype=float)
        return self.finalize_signals(positions, dates, prices)

    def get_latest_signal(self):
        return {'position': 0.0}
""".strip()


def _reset_strategy_modules() -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _write_engine_project(
    root: Path,
    *,
    engine_name: str,
    engine_code: str,
    extra_yaml: str = "",
    primary_csv: str | None = None,
) -> None:
    _reset_strategy_modules()
    (root / "strategies").mkdir()
    (root / "strategies" / "__init__.py").write_text("", encoding="utf-8")
    strategy_dir = root / "strategies" / engine_name
    strategy_dir.mkdir()
    (strategy_dir / "__init__.py").write_text("", encoding="utf-8")
    (strategy_dir / "engine.py").write_text(engine_code, encoding="utf-8")
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "ethusd.csv").write_text(
        primary_csv
        or (
            "timestamp,close\n"
            "2026-01-01T00:00:00Z,100\n"
            "2026-01-02T00:00:00Z,110\n"
            "2026-01-03T00:00:00Z,120\n"
        ),
        encoding="utf-8",
    )
    yaml = f"""
settings:
  price_data:
    default_adapter: csv
    default_timeframe: 1d
strategies:
  - id: {engine_name}
    name: "{engine_name}"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.{engine_name}.engine
    trade_log: data/trade_log_{engine_name}.csv
    price_data:
      adapter: csv
      path: data/ethusd.csv
{extra_yaml}
""".strip() + "\n"
    (root / "strategies.yaml").write_text(yaml, encoding="utf-8")


def test_load_config_synthesizes_primary_feed_and_default_contract(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings: {}
strategies:
  - id: demo
    name: Demo
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.demo.engine
    trade_log: data/demo.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg["settings"]["data_contract"]["profile"] == "daily"
    strategy = cfg["strategies"][0]
    assert strategy["_data_contract"]["profile"] == "daily"
    assert strategy["_feeds"]["primary"]["kind"] == "bars"
    assert strategy["_feeds"]["primary"]["adapter"] == "abel"
    assert strategy["_feeds"]["primary"]["symbol"] == "ETHUSD"


def test_load_config_rejects_user_defined_primary_feed(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings: {}
strategies:
  - id: demo
    name: Demo
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.demo.engine
    trade_log: data/demo.csv
    feeds:
      primary:
        kind: series
        adapter: csv
        path: data/feed.csv
        field: value
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feeds.primary is reserved"):
        load_config(config_path)


def test_align_series_rejects_naive_auxiliary_dates():
    aux = pd.Series([1.0, 2.0], index=pd.date_range("2026-01-01", periods=2))
    dates = pd.date_range("2026-01-01", periods=2, tz="UTC")

    with pytest.raises(FeedAlignmentError, match="UTC-aware"):
        align_series_to_dates(aux, dates, profile="daily")


def test_validate_signal_output_rejects_unsorted_dates():
    with pytest.raises(SignalContractError, match="strictly increasing"):
        validate_signal_output(
            [0.0, 1.0],
            pd.to_datetime(["2026-01-02T00:00:00Z", "2026-01-01T00:00:00Z"], utc=True),
            [100.0, 110.0],
        )


def test_validate_signal_output_rejects_mismatched_lengths():
    with pytest.raises(SignalContractError, match="identical lengths"):
        validate_signal_output(
            [0.0, 1.0],
            pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True),
            [100.0, 110.0],
        )


def test_run_fails_early_on_naive_signal_dates(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_engine_project(root, engine_name="naive_dates", engine_code=NAIVE_ENGINE_CODE)
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "naive_dates"])

        assert result.exit_code != 0
        assert "NaiveDatesEngine" in result.output
        assert "UTC-aware" in result.output
        sys.path.pop(0)
        _reset_strategy_modules()


def test_paper_fails_early_on_naive_signal_dates(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_engine_project(root, engine_name="naive_dates", engine_code=NAIVE_ENGINE_CODE)
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["paper", "--strategy", "naive_dates"])

        assert result.exit_code != 0
        assert "NaiveDatesEngine" in result.output
        assert "UTC-aware" in result.output
        sys.path.pop(0)
        _reset_strategy_modules()


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
        _write_engine_project(root, engine_name="feed_demo", engine_code=FEED_ENGINE_CODE, extra_yaml=extra_yaml)
        sys.path.insert(0, str(root))

        result = runner.invoke(main, ["run", "--strategy", "feed_demo"])

        assert result.exit_code == 0, result.output
        trade_df = read_trade_log("data/trade_log_feed_demo.csv")
        assert list(trade_df["position"].round(2)) == [0.2, 0.4, 0.6]
        sys.path.pop(0)
        _reset_strategy_modules()


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
        _write_engine_project(
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
        _reset_strategy_modules()


def test_run_supports_naive_primary_feed_timestamps_from_csv_loader(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_engine_project(
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
        _reset_strategy_modules()


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
        _write_engine_project(
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
        _reset_strategy_modules()


def test_run_fails_early_on_undeclared_feed_access(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_engine_project(
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
        _reset_strategy_modules()


def test_paper_fails_early_on_undeclared_feed_access(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        _write_engine_project(
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
        _reset_strategy_modules()
