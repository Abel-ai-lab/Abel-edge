"""Adapter registry tests for config-driven third-party feed adapters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import importlib
import json
import sys
from pathlib import Path

from click.testing import CliRunner
import pandas as pd
import pytest

from abel_edge.cli import main
from abel_edge.config import load_config
from abel_edge.engine.adapter_registry import AbelDataFeedAdapter, FeedLoadRequest
from abel_edge.engine.cache import cache_entry_for_request, load_cached_metadata, write_cached_bars
from abel_edge.engine.ledger import read_trade_log


ENGINE_CODE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.base import StrategyEngine


class AdapterRegistryDemoEngine(StrategyEngine):
    def compute_signals(self):
        bars = self.load_bars(limit=3)
        target = bars[bars["symbol"] == self.context.get("asset", "ETHUSD")].copy().sort_values("timestamp")
        dates = pd.DatetimeIndex(target["timestamp"])
        prices = target["close"].astype(float).to_numpy()
        scale = self.feed_series("risk_scale", align_to=dates, method="ffill", allow_gaps=False)
        positions = scale.astype(float).to_numpy()
        return self.finalize_signals(positions, dates, prices)

    def get_latest_signal(self):
        return {"position": 0.0}
""".strip()


ADAPTER_MODULE = """
from __future__ import annotations

import pandas as pd

from abel_edge.engine.adapter_registry import register_adapter


class ConstantSeriesAdapter:
    assume_utc_for_naive = False

    def load(self, request):
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                ],
                "value": [0.2, 0.4, 0.6],
            }
        )


register_adapter("constant_series", ConstantSeriesAdapter())
""".strip()


def _reset_modules() -> None:
    for name in list(sys.modules):
        if name in {"project_adapters"} or name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def test_load_config_rejects_missing_declared_adapter(tmp_path):
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
      risk_scale:
        kind: series
        adapter: missing_registry_adapter
        field: value
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not registered"):
        load_config(config_path)


def test_load_config_imports_project_local_adapter_modules(tmp_path, monkeypatch):
    adapter_path = tmp_path / "project_adapters.py"
    adapter_path.write_text(ADAPTER_MODULE + "\n", encoding="utf-8")
    config_path = tmp_path / "strategies.yaml"
    config_path.write_text(
        """
settings:
  data_adapters:
    imports:
      - project_adapters
strategies:
  - id: demo
    name: Demo
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.demo.engine
    trade_log: data/demo.csv
    feeds:
      risk_scale:
        kind: series
        adapter: constant_series
        field: value
""".strip()
        + "\n",
        encoding="utf-8",
    )

    _reset_modules()
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = load_config(config_path)

    assert cfg["strategies"][0]["_feeds"]["risk_scale"]["adapter"] == "constant_series"


def test_run_supports_project_local_series_adapter_via_registry(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        root = Path.cwd()
        (root / "project_adapters.py").write_text(ADAPTER_MODULE + "\n", encoding="utf-8")
        (root / "strategies").mkdir()
        (root / "strategies" / "__init__.py").write_text("", encoding="utf-8")
        (root / "strategies" / "adapter_registry_demo").mkdir()
        (root / "strategies" / "adapter_registry_demo" / "__init__.py").write_text(
            "", encoding="utf-8"
        )
        (root / "strategies" / "adapter_registry_demo" / "engine.py").write_text(
            ENGINE_CODE + "\n",
            encoding="utf-8",
        )
        (root / "data").mkdir()
        (root / "data" / "ethusd.csv").write_text(
            "timestamp,close\n"
            "2026-01-01,100\n"
            "2026-01-02,110\n"
            "2026-01-03,120\n",
            encoding="utf-8",
        )
        (root / "strategies.yaml").write_text(
            """
settings:
  price_data:
    default_adapter: csv
    default_timeframe: 1d
  data_adapters:
    imports:
      - project_adapters
strategies:
  - id: adapter_registry_demo
    name: "Adapter Registry Demo"
    asset: ETHUSD
    color: "#2563EB"
    engine: strategies.adapter_registry_demo.engine
    trade_log: data/trade_log_adapter_registry_demo.csv
    price_data:
      adapter: csv
      path: data/ethusd.csv
    feeds:
      risk_scale:
        kind: series
        adapter: constant_series
        field: value
""".strip()
            + "\n",
            encoding="utf-8",
        )

        _reset_modules()
        sys.path.insert(0, str(root))
        try:
            result = runner.invoke(main, ["run", "--strategy", "adapter_registry_demo"])
            assert result.exit_code == 0, result.output
            trade_df = read_trade_log("data/trade_log_adapter_registry_demo.csv")
            assert list(trade_df["position"].round(2)) == [0.2, 0.4, 0.6]
        finally:
            sys.path.pop(0)
            _reset_modules()


def test_abel_bars_adapter_refreshes_close_only_cache_with_full_ohlcv(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_bars(*, symbols, start=None, end=None, timeframe="1d", limit=None, fields=None, config=None):
        calls.append(
            {
                "symbols": symbols,
                "start": start,
                "end": end,
                "timeframe": timeframe,
                "limit": limit,
                "fields": fields,
                "config": config,
            }
        )
        return pd.DataFrame(
            {
                "timestamp": ["2020-01-02T00:00:00Z"],
                "symbol": ["AAPL"],
                "open": [99.0],
                "high": [101.0],
                "low": [98.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        )

    import abel_edge.plugins.abel.prices as prices_module

    monkeypatch.setattr(prices_module, "fetch_bars", fake_fetch_bars)
    request = FeedLoadRequest(
        adapter="abel",
        kind="bars",
        symbol="AAPL",
        field=None,
        timeframe="1d",
        start="2020-01-01",
        end=None,
        limit=10,
        profile="daily",
        options={"fields": ["close"], "cache_root": str(tmp_path)},
        strategy_id=None,
        feed_name="primary",
    )
    entry = cache_entry_for_request(
        adapter="abel",
        symbol="AAPL",
        timeframe="1d",
        profile="daily",
        options=request.options,
        cache_root=tmp_path,
    )
    write_cached_bars(
        entry,
        pd.DataFrame(
            {
                "timestamp": ["2020-01-02T00:00:00Z"],
                "symbol": ["AAPL"],
                "close": [100.0],
            }
        ),
        requested_start="2020-01-01",
    )

    frame = AbelDataFeedAdapter().load(request)
    metadata = load_cached_metadata(entry)

    assert calls[0]["fields"] == ["open", "high", "low", "close", "volume"]
    assert calls[0]["limit"] == 10
    assert list(frame.columns) == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    assert metadata["columns"] == ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    assert metadata["requested_range"]["limit"] == 10


def test_abel_bars_adapter_refreshes_stale_cache_confirmation(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_bars(*, symbols, start=None, end=None, timeframe="1d", limit=None, fields=None, config=None):
        calls.append({"symbols": symbols, "fields": fields})
        return pd.DataFrame(
            {
                "timestamp": ["2020-01-03T00:00:00Z"],
                "symbol": ["AAPL"],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1100.0],
            }
        )

    import abel_edge.plugins.abel.prices as prices_module

    monkeypatch.setattr(prices_module, "fetch_bars", fake_fetch_bars)
    request = FeedLoadRequest(
        adapter="abel",
        kind="bars",
        symbol="AAPL",
        field=None,
        timeframe="1d",
        start="2020-01-01",
        end=None,
        limit=10,
        profile="daily",
        options={"cache_root": str(tmp_path), "max_cache_age_seconds": 1},
        strategy_id=None,
        feed_name="primary",
    )
    entry = cache_entry_for_request(
        adapter="abel",
        symbol="AAPL",
        timeframe="1d",
        profile="daily",
        options=request.options,
        cache_root=tmp_path,
    )
    metadata = write_cached_bars(
        entry,
        pd.DataFrame(
            {
                "timestamp": ["2020-01-02T00:00:00Z"],
                "symbol": ["AAPL"],
                "open": [99.0],
                "high": [101.0],
                "low": [98.0],
                "close": [100.0],
                "volume": [1000.0],
            }
        ),
        requested_start="2020-01-01",
    )
    metadata["updated_at"] = (datetime.now(tz=UTC) - timedelta(days=1)).isoformat()
    entry.meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    frame = AbelDataFeedAdapter().load(request)

    assert calls
    assert float(frame.iloc[-1]["close"]) == 101.0
