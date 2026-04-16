"""Shared helpers for data-contract runtime tests."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


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


def reset_strategy_modules() -> None:
    for name in list(sys.modules):
        if name == "strategies" or name.startswith("strategies."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def write_engine_project(
    root: Path,
    *,
    engine_name: str,
    engine_code: str,
    extra_yaml: str = "",
    primary_csv: str | None = None,
) -> None:
    reset_strategy_modules()
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
