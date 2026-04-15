"""Project scaffolding for `causal-edge init`."""

from __future__ import annotations

import shutil
from pathlib import Path


def scaffold_project(name: str) -> Path:
    """Create a new causal-edge project directory.

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

    # Create directory structure
    root.mkdir()
    (root / "strategies" / "sma_crossover").mkdir(parents=True)
    (root / "strategies" / "momentum_ml").mkdir(parents=True)
    (root / "strategies" / "causal_demo").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data" / ".gitkeep").write_text("", encoding="utf-8")

    # Copy example engines + causal graph data
    examples_dir = Path(__file__).parent.parent / "examples"
    for name_dir in ("sma_crossover", "momentum_ml", "causal_demo"):
        src = examples_dir / name_dir / "engine.py"
        dst = root / "strategies" / name_dir / "engine.py"
        if src.exists():
            shutil.copy2(src, dst)

    # Copy causal graph JSON
    graph_src = examples_dir / "causal_demo" / "causal_graph.json"
    if graph_src.exists():
        shutil.copy2(graph_src, root / "strategies" / "causal_demo" / "causal_graph.json")

    # Fallback if examples not found (pip install without source tree)
    if not (root / "strategies" / "sma_crossover" / "engine.py").exists():
        (root / "strategies" / "sma_crossover" / "engine.py").write_text(
            _SMA_ENGINE_SRC,
            encoding="utf-8",
        )

    if not (root / "strategies" / "causal_demo" / "causal_graph.json").exists():
        (root / "strategies" / "causal_demo" / "causal_graph.json").write_text(
            _CAUSAL_GRAPH_JSON,
            encoding="utf-8",
        )

    (root / "strategies" / "__init__.py").write_text("", encoding="utf-8")
    (root / "strategies" / "sma_crossover" / "__init__.py").write_text("", encoding="utf-8")
    (root / "strategies" / "momentum_ml" / "__init__.py").write_text("", encoding="utf-8")
    (root / "strategies" / "causal_demo" / "__init__.py").write_text("", encoding="utf-8")

    # strategies.yaml
    (root / "strategies.yaml").write_text(_STRATEGIES_YAML, encoding="utf-8")

    # .env.example
    (root / ".env.example").write_text(_ENV_EXAMPLE, encoding="utf-8")

    # CLAUDE.md
    (root / "CLAUDE.md").write_text(_CLAUDE_MD, encoding="utf-8")

    # AGENTS.md
    (root / "AGENTS.md").write_text(_AGENTS_MD, encoding="utf-8")

    return root


_STRATEGIES_YAML = """\
# causal-edge project configuration
# Run: causal-edge run && causal-edge dashboard && causal-edge validate

settings:
  capital: 100000
  port: 8080
  theme: dark
  price_data:
    default_source: abel
    default_timeframe: 1d

strategies:
  - id: sma_crossover
    name: "SMA Crossover"
    asset: DEMO
    color: "#0A84FF"
    engine: strategies.sma_crossover.engine
    trade_log: "data/trade_log_sma_crossover.csv"
    paper_log: "data/paper_log_sma_crossover.csv"

  - id: momentum_ml
    name: "Momentum ML"
    asset: DEMO
    color: "#FF9500"
    engine: strategies.momentum_ml.engine
    trade_log: "data/trade_log_momentum_ml.csv"
    paper_log: "data/paper_log_momentum_ml.csv"

  - id: causal_demo
    name: "Causal Voting (TON)"
    asset: TON
    color: "#30D158"
    engine: strategies.causal_demo.engine
    trade_log: "data/trade_log_causal_demo.csv"
    paper_log: "data/paper_log_causal_demo.csv"
"""

_ENV_EXAMPLE = """\
# Abel CAP API key (optional — for causal discovery and live price data)
# Get one at https://abel.ai
# ABEL_API_KEY=your_key_here

# Optional: override the public auth base URL for `causal-edge login`
# ABEL_AUTH_BASE_URL=https://api.abel.ai/echo

# Optional: override the public CAP base URL
# ABEL_CAP_BASE_URL=https://cap.abel.ai/api

# Optional: point real-price strategies at a local CSV instead of Abel
# PRICE_DATA_SOURCE=csv
"""

_CLAUDE_MD = """\
# CLAUDE.md — project harness

## Constraints
- strategies.yaml is the single source of truth
- All features must use shift(1) — zero look-ahead tolerance
- strategies/ must not import causal_edge internals (except engine base)

## Commands
causal-edge run         # run strategies, write trade logs
causal-edge dashboard   # generate dashboard.html
causal-edge validate    # Abel Proof audited validation
causal-edge status      # show strategy summary
"""

_AGENTS_MD = """\
# Project — Agent Entry Point

## I want to...

### Run everything
    causal-edge run && causal-edge dashboard && causal-edge validate

### Add a strategy
1. Create strategies/my_strategy/engine.py implementing StrategyEngine
2. Add entry to strategies.yaml
3. Run: causal-edge run --strategy my_strategy
4. Run: causal-edge validate --strategy my_strategy

### Fix a failing validation
    causal-edge validate --verbose
See the failure→fix mapping in the causal-edge docs.

### View the dashboard
    causal-edge dashboard && open dashboard.html
"""

_SMA_ENGINE_SRC = """\
from __future__ import annotations

import numpy as np
import pandas as pd

from causal_edge.engine.base import StrategyEngine


class SMAEngine(StrategyEngine):
    def __init__(self, context=None, n_days=500):
        super().__init__(context=context)
        self.n_days = n_days
        self.fast = 10
        self.slow = 30

    def compute_signals(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.02, self.n_days)
        prices = 100.0 * np.cumprod(1.0 + returns)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=self.n_days)
        fast_ma = pd.Series(prices).rolling(self.fast).mean().shift(1).values
        slow_ma = pd.Series(prices).rolling(self.slow).mean().shift(1).values
        positions = np.where(fast_ma > slow_ma, 1.0, 0.0)
        positions[:self.slow + 1] = 0.0
        return positions, dates, prices

    def get_latest_signal(self):
        positions, dates, prices = self.compute_signals()
        return {"position": float(positions[-1]), "date": str(dates[-1].date())}
"""

_CAUSAL_GRAPH_JSON = """\
{
  "target": "TONUSD",
  "source": "Abel Causal Graph (cap.abel.ai)",
  "description": "Causal neighborhood of TONUSD - 5 equity parents and 3 children discovered via Abel's causal graph API. Demo defaults use lag=1 and window=5 unless manually overridden.",
  "parents": [
    {"ticker": "GBLI", "field": "price", "type": "parent"},
    {"ticker": "HSON", "field": "price", "type": "parent"},
    {"ticker": "SITC", "field": "price", "type": "parent"},
    {"ticker": "EVC", "field": "price", "type": "parent"},
    {"ticker": "EAI", "field": "price", "type": "parent"}
  ],
  "children": [
    {"ticker": "ESBA", "field": "price", "type": "child", "lag": 2},
    {"ticker": "SIRI", "field": "price", "type": "child", "lag": 3},
    {"ticker": "TVC", "field": "price", "type": "child", "lag": 2, "window": 8}
  ],
  "note": "To discover causal parents for other assets, use Abel API: causal-edge discover <TICKER>"
}
"""
