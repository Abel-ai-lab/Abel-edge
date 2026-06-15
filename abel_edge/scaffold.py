"""Project scaffolding for `abel-edge init`."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd


def scaffold_project(name: str) -> Path:
    """Create a new abel-edge project directory.

    Args:
        name: Project directory name (created in current working directory)

    Returns:
        Path to created directory

    Raises:
        FileExistsError: If directory already exists
    """
    root = Path(name)
    if root.exists():
        raise FileExistsError(
            f"Directory '{name}' already exists. "
            f"Choose a different name or remove the existing directory."
        )

    root.mkdir()
    strategy_names = ("sma_crossover", "momentum_ml", "feed_overlay_demo")
    for strategy_name in strategy_names:
        (root / "strategies" / strategy_name).mkdir(parents=True)
    (root / "data").mkdir()
    _write_demo_data(root / "data")

    examples_dir = Path(__file__).parent.parent / "examples"
    for strategy_name in strategy_names:
        src = examples_dir / strategy_name / "engine.py"
        dst = root / "strategies" / strategy_name / "engine.py"
        if src.exists():
            shutil.copy2(src, dst)

    _write_if_missing(root / "strategies" / "sma_crossover" / "engine.py", _SMA_ENGINE_SRC)
    _write_if_missing(root / "strategies" / "momentum_ml" / "engine.py", _MOMENTUM_ML_ENGINE_SRC)
    _write_if_missing(
        root / "strategies" / "feed_overlay_demo" / "engine.py",
        _FEED_OVERLAY_ENGINE_SRC,
    )

    (root / "strategies" / "__init__.py").write_text("", encoding="utf-8")
    for strategy_name in strategy_names:
        (root / "strategies" / strategy_name / "__init__.py").write_text("", encoding="utf-8")

    (root / "strategies.yaml").write_text(_STRATEGIES_YAML, encoding="utf-8")
    (root / ".env.example").write_text(_ENV_EXAMPLE, encoding="utf-8")
    (root / "CLAUDE.md").write_text(_CLAUDE_MD, encoding="utf-8")
    (root / "AGENTS.md").write_text(_AGENTS_MD, encoding="utf-8")

    return root


def _write_demo_data(data_dir: Path) -> None:
    dates = pd.date_range("2025-01-01", periods=180, freq="B")
    target_rows = ["timestamp,close"]
    driver_rows = ["timestamp,close"]
    scale_rows = ["timestamp,value"]

    for i, ts in enumerate(dates):
        target_close = 100.0 + 0.18 * i + 3.4 * math.sin(i / 6.0) + 1.6 * math.cos(i / 11.0)
        driver_close = 78.0 + 0.14 * i + 2.6 * math.sin((i - 2) / 7.0) + 1.2 * math.cos(i / 13.0)
        risk_scale = 0.55 + 0.25 * math.sin(i / 9.0)
        target_rows.append(f"{ts.date().isoformat()},{target_close:.4f}")
        driver_rows.append(f"{ts.date().isoformat()},{driver_close:.4f}")
        scale_rows.append(f"{ts.date().isoformat()},{max(0.15, min(1.0, risk_scale)):.4f}")

    (data_dir / "demo_target.csv").write_text("\n".join(target_rows) + "\n", encoding="utf-8")
    (data_dir / "demo_driver.csv").write_text("\n".join(driver_rows) + "\n", encoding="utf-8")
    (data_dir / "demo_scale.csv").write_text("\n".join(scale_rows) + "\n", encoding="utf-8")


_STRATEGIES_YAML = """\
# abel-edge project configuration
# Standalone abel-edge project with local sample-data strategies.
# Run: abel-edge run && abel-edge validate

settings:
  capital: 100000
  price_data:
    default_adapter: csv
    default_timeframe: 1d

strategies:
  - id: sma_crossover
    name: "SMA Crossover"
    asset: DEMO
    color: "#0A84FF"
    engine: strategies.sma_crossover.engine
    trade_log: "data/trade_log_sma_crossover.csv"
    paper_log: "data/paper_log_sma_crossover.csv"
    price_data:
      adapter: csv
      symbol: DEMO
      path: data/demo_target.csv

  - id: momentum_ml
    name: "Momentum ML"
    asset: DEMO
    color: "#FF9500"
    engine: strategies.momentum_ml.engine
    trade_log: "data/trade_log_momentum_ml.csv"
    paper_log: "data/paper_log_momentum_ml.csv"
    price_data:
      adapter: csv
      symbol: DEMO
      path: data/demo_target.csv

  - id: feed_overlay_demo
    name: "Feed Overlay Demo"
    asset: DEMO
    color: "#30D158"
    engine: strategies.feed_overlay_demo.engine
    trade_log: "data/trade_log_feed_overlay_demo.csv"
    paper_log: "data/paper_log_feed_overlay_demo.csv"
    price_data:
      adapter: csv
      symbol: DEMO
      path: data/demo_target.csv
    feeds:
      btc_ref:
        kind: bars
        adapter: csv
        path: data/demo_driver.csv
        symbol: DEMO_DRV
      risk_scale:
        kind: series
        adapter: csv
        path: data/demo_scale.csv
        field: value
"""

_ENV_EXAMPLE = """\
# Abel CAP API key (optional — for causal discovery and live price data)
# Get one at https://abel.ai
# ABEL_API_KEY=your_key_here

# Optional: override the public auth base URL for `abel-edge login`
# ABEL_AUTH_BASE_URL=https://api.abel.ai/echo

# Optional: override the public CAP base URL
# ABEL_CAP_BASE_URL=https://cap.abel.ai/api
"""

_CLAUDE_MD = """\
# CLAUDE.md — project harness

## Constraints
- strategies.yaml is the single source of truth
- implement `compute_decisions(self, ctx)` for new strategies
- read market data through `DecisionContext`
- return `ctx.decisions(next_position)`
- use `abel-edge evaluate --workdir strategies/<id>` when you need semantic evidence

## Commands
abel-edge run         # run strategies, write trade logs
abel-edge validate    # Abel Proof audited validation
abel-edge status      # show strategy summary
"""

_AGENTS_MD = """\
# Project — Agent Entry Point

This is a standalone `abel-edge` project scaffold with local sample data.
It is not an Abel-alpha branch workspace.

## I want to...

### Run everything
    abel-edge run && abel-edge validate

### Add a strategy
1. Create `strategies/my_strategy/engine.py`
2. Implement `compute_decisions(self, ctx)`
3. Add an entry to `strategies.yaml`
4. Run `abel-edge run --strategy my_strategy`
5. Run `abel-edge validate --strategy my_strategy`

### Inspect runtime semantics for one strategy
    abel-edge debug-evaluate --workdir strategies/my_strategy

## Authoring surface

- primary target data: `ctx.target.series("close")`
- declared auxiliary feeds: `ctx.feed(name).native_series(...)` or `ctx.feed(name).asof_series(...)`
- point inspection: `ctx.points()`
- strategy output: `ctx.decisions(next_position)`
"""

_SMA_ENGINE_SRC = """\
from __future__ import annotations

from abel_edge.engine.base import StrategyEngine


class SMAEngine(StrategyEngine):
    def __init__(self, context=None):
        super().__init__(context=context)
        self.fast = 10
        self.slow = 30

    def compute_decisions(self, ctx):
        close = ctx.target.series("close")
        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()
        next_position = (fast_ma > slow_ma).astype(float).fillna(0.0)
        if len(next_position) > 0:
            next_position.iloc[: self.slow] = 0.0
        return ctx.decisions(next_position)
"""

_MOMENTUM_ML_ENGINE_SRC = """\
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from abel_edge.engine.base import StrategyEngine


class MomentumMLEngine(StrategyEngine):
    def __init__(self, context=None):
        super().__init__(context=context)
        self.train_window = 126
        self.retrain_every = 5

    def compute_decisions(self, ctx):
        close = ctx.target.series("close").astype(float)
        returns = close.pct_change().fillna(0.0)
        features = pd.DataFrame(
            {
                "ret_1d": returns,
                "ret_5d": returns.rolling(5, min_periods=5).sum(),
                "ret_20d": returns.rolling(20, min_periods=20).sum(),
                "vol_20d": returns.rolling(20, min_periods=20).std(),
                "sma_gap_10": close / close.rolling(10, min_periods=10).mean() - 1.0,
                "rsi_14": _rsi(returns, 14),
            },
            index=close.index,
        )
        target = (returns.shift(-1) > 0).astype(int)

        next_position = pd.Series(0.0, index=close.index, dtype=float)
        start = max(self.train_window, 25)
        last_model = None
        last_train_day = 0

        for t in range(start, len(close)):
            if last_model is None or (t - last_train_day) >= self.retrain_every:
                train_start = max(0, t - self.train_window)
                train_slice = features.iloc[train_start:t]
                target_slice = target.iloc[train_start:t]
                valid = (~train_slice.isna().any(axis=1)) & target_slice.notna()
                if int(valid.sum()) < 30:
                    continue
                X_train = train_slice.loc[valid].to_numpy()
                y_train = target_slice.loc[valid].to_numpy()
                if len(np.unique(y_train)) < 2:
                    continue
                model = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                last_model = model
                last_train_day = t

            x_t = features.iloc[t].to_numpy(dtype=float).reshape(1, -1)
            if np.isnan(x_t).any():
                continue
            prob = last_model.predict_proba(x_t)[0]
            next_position.iloc[t] = 1.0 if prob[1] > 0.55 else 0.0

        return ctx.decisions(next_position)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
"""

_FEED_OVERLAY_ENGINE_SRC = """\
from __future__ import annotations

from abel_edge.engine.base import StrategyEngine


class FeedOverlayDemoEngine(StrategyEngine):
    def compute_decisions(self, ctx):
        btc_close = ctx.feed("btc_ref").asof_series("close").astype(float)
        risk_scale = ctx.feed("risk_scale").asof_series("value").astype(float)
        btc_trend = (btc_close > btc_close.rolling(2, min_periods=2).mean()).astype(float)
        next_position = (risk_scale * btc_trend.fillna(0.0)).clip(lower=0.0, upper=1.0)
        return ctx.decisions(next_position)
"""


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")
