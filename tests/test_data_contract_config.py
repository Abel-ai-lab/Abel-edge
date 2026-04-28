"""Config and pure-contract tests for the data-feed runtime model."""

from __future__ import annotations

import pandas as pd
import pytest

from abel_edge.config import load_config
from abel_edge.engine.base import StrategyEngine
from abel_edge.engine.feed_contract import FeedAlignmentError, align_series_to_dates
from abel_edge.engine.signal_contract import SignalContractError, validate_signal_output


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


def test_load_config_allows_primary_symbol_override_in_price_data(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings: {}
strategies:
  - id: demo
    name: Demo
    asset: TON
    color: "#2563EB"
    engine: strategies.demo.engine
    trade_log: data/demo.csv
    price_data:
      adapter: csv
      symbol: TONUSD
      path: data/tonusd.csv
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg["strategies"][0]["_feeds"]["primary"]["symbol"] == "TONUSD"


def test_load_config_preserves_primary_feed_extra_options_from_price_data(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings: {}
strategies:
  - id: demo
    name: Demo
    asset: ETH
    color: "#2563EB"
    engine: strategies.demo.engine
    trade_log: data/demo.csv
    price_data:
      adapter: csv
      symbol: ETHUSD
      path: data/ethusd.csv
      adjusted: false
      backend: fmp
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    primary = cfg["strategies"][0]["_feeds"]["primary"]
    assert primary["symbol"] == "ETHUSD"
    assert primary["adjusted"] is False
    assert primary["backend"] == "fmp"


def test_load_config_preserves_multi_symbol_bars_feeds_without_asset_fallback(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings: {}
strategies:
  - id: demo
    name: Demo
    asset: TON
    color: "#2563EB"
    engine: strategies.demo.engine
    trade_log: data/demo.csv
    feeds:
      parents_bars:
        kind: bars
        adapter: csv
        path: data/parents.csv
        symbols:
          - WTM
          - SPY
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert "symbol" not in cfg["strategies"][0]["_feeds"]["parents_bars"]


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


def test_bind_price_loader_fails_with_migration_guidance():
    class DummyEngine(StrategyEngine):
        def compute_signals(self):
            raise NotImplementedError

        def get_latest_signal(self):
            return {"position": 0.0}

    engine = DummyEngine()

    with pytest.raises(RuntimeError, match="deprecated and no longer supported"):
        engine.bind_price_loader(object(), {})
