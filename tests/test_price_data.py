"""Tests for price data source resolution and normalization."""

from pathlib import Path

import pandas as pd
import pytest

from causal_edge import config as config_module
from causal_edge.engine.feed_contract import FeedNormalizationError
from causal_edge.engine import price_data as price_data_module


def test_normalize_bars_maps_aliases_and_sorts():
    df = pd.DataFrame(
        {
            "date": ["2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"],
            "symbol": ["ETHUSD", "ETHUSD"],
            "price": [2.0, 1.0],
        }
    )
    bars = price_data_module.normalize_bars(df)
    assert list(bars.columns[:3]) == ["timestamp", "symbol", "close"]
    assert list(bars["close"]) == [1.0, 2.0]


def test_load_bars_from_csv_fills_single_symbol(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"], "close": [1.0]}).to_csv(path, index=False)
    bars = price_data_module.load_bars_from_csv(path, symbols=["ETHUSD"])
    assert list(bars["symbol"]) == ["ETHUSD"]


def test_load_bars_from_csv_requires_symbol_for_multi_symbol(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"], "close": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="include 'symbol'"):
        price_data_module.load_bars_from_csv(path, symbols=["ETHUSD", "BTCUSD"])


def test_normalize_bars_rejects_naive_daily_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": ["2025-01-01", "2025-01-02"],
            "symbol": ["ETHUSD", "ETHUSD"],
            "close": [1.0, 2.0],
        }
    )

    with pytest.raises(FeedNormalizationError, match="UTC-aware"):
        price_data_module.normalize_bars(df)


def test_normalize_bars_rejects_duplicate_symbol_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "symbol": ["ETHUSD", "ETHUSD"],
            "close": [1.0, 2.0],
        }
    )

    with pytest.raises(FeedNormalizationError, match="duplicate timestamps"):
        price_data_module.normalize_bars(df)


def test_resolve_price_config_uses_strategy_override():
    settings = {"price_data": {"default_source": "abel", "default_timeframe": "1d"}}
    strategy = {"asset": "ETHUSD", "price_data": {"source": "csv", "path": "data/prices.csv"}}
    cfg = price_data_module.resolve_price_config(settings, strategy)
    assert cfg["source"] == "csv"
    assert cfg["symbol"] == "ETHUSD"


def test_load_config_validates_csv_path(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings:
  price_data:
    default_source: abel
strategies:
  - id: test
    name: Test
    asset: ETHUSD
    color: '#123456'
    engine: strategies.ethusd_causal.engine
    trade_log: data/test.csv
    price_data:
      source: csv
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="price_data.path"):
        config_module.load_config(config_path)


def test_load_config_applies_execution_defaults(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings: {}
strategies:
  - id: test
    name: Test
    asset: ETHUSD
    color: '#123456'
    engine: strategies.ethusd_causal.engine
    trade_log: data/test.csv
""",
        encoding="utf-8",
    )

    cfg = config_module.load_config(config_path)

    assert cfg["settings"]["execution"]["cost_bps"] == 0
    assert cfg["settings"]["execution"]["max_abs_position"] is None


def test_load_config_validates_execution_settings(tmp_path):
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings:
  execution:
    cost_bps: -1
strategies:
  - id: test
    name: Test
    asset: ETHUSD
    color: '#123456'
    engine: strategies.ethusd_causal.engine
    trade_log: data/test.csv
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="execution.cost_bps"):
        config_module.load_config(config_path)


def test_load_config_prefers_local_overlay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("strategies.yaml").write_text(
        """
settings:
  theme: dark
strategies: []
""",
        encoding="utf-8",
    )
    Path("strategies.local.yaml").write_text(
        """
settings:
  theme: light
strategies:
  - id: local
    name: Local Strategy
    asset: ETHUSD
    color: '#123456'
    engine: strategies.local.engine
    trade_log: data/local.csv
""",
        encoding="utf-8",
    )

    cfg = config_module.load_config()

    assert cfg["settings"]["theme"] == "light"
    assert [strategy["id"] for strategy in cfg["strategies"]] == ["local"]


def test_load_config_explicit_path_overrides_local_overlay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    explicit_path = Path("custom.yaml")
    Path("strategies.local.yaml").write_text(
        """
settings: {}
strategies:
  - id: local
    name: Local Strategy
    asset: ETHUSD
    color: '#123456'
    engine: strategies.local.engine
    trade_log: data/local.csv
""",
        encoding="utf-8",
    )
    explicit_path.write_text(
        """
settings: {}
strategies:
  - id: explicit
    name: Explicit Strategy
    asset: BTCUSD
    color: '#654321'
    engine: strategies.explicit.engine
    trade_log: data/explicit.csv
""",
        encoding="utf-8",
    )

    cfg = config_module.load_config(explicit_path)

    assert [strategy["id"] for strategy in cfg["strategies"]] == ["explicit"]
