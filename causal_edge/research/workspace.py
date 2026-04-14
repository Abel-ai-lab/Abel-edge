"""Research workspace initialization helpers."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from causal_edge.plugins.abel.client import AbelClient, normalize_public_node_id
from causal_edge.plugins.abel.credentials import resolve_api_key
from causal_edge.research.constants import RESULTS_HEADER

STRATEGY_TEMPLATE = '''"""Strategy for {ticker} - experiment baseline.

Fill in run_strategy(). Everything else is handled by causal-edge.
Run: python -m causal_edge.research.evaluate --workdir .
"""
import numpy as np
import pandas as pd


def run_strategy():
    """Your strategy logic. Returns (pnl, dates, positions).

    pnl: np.ndarray of daily strategy returns
    dates: pd.DatetimeIndex
    positions: np.ndarray of daily position sizes (0=flat, 1=long)
    """
    raise NotImplementedError("Fill in run_strategy()")
'''

MEMORY_TEMPLATE = """# {ticker} Research Memory

## K Budget
- Discovery: K=? (fill after discovery)

## Baseline
- (none yet)

## Exhausted Directions

## What Worked

## Ideas Not Yet Tried
"""

def init_workspace(ticker: str, workdir: Path | str | None = None) -> Path:
    ticker = ticker.upper()
    workspace = Path(workdir) if workdir is not None else Path("research") / ticker.lower()
    workspace.mkdir(parents=True, exist_ok=True)

    _write_if_missing(workspace / "strategy.py", STRATEGY_TEMPLATE.format(ticker=ticker))
    _write_if_missing(workspace / "results.tsv", RESULTS_HEADER)
    _write_if_missing(workspace / "memory.md", MEMORY_TEMPLATE.format(ticker=ticker))

    discovery_path = workspace / "discovery.json"
    if not discovery_path.exists():
        discovery = _try_abel_discovery(ticker)
        discovery_path.write_text(json.dumps(discovery, indent=2), encoding="utf-8")

    return workspace


def _try_abel_discovery(ticker: str) -> dict:
    api_key = _resolve_workspace_api_key()
    if not api_key:
        return {
            "ticker": ticker,
            "source": "template (no ABEL_API_KEY)",
            "parents": [],
            "blanket_new": [],
            "children": [],
            "K_discovery": 0,
            "note": "Set ABEL_API_KEY or run causal-edge discover manually.",
        }

    try:
        client = AbelClient()
        node_id = normalize_public_node_id(ticker)
        with ThreadPoolExecutor(max_workers=2) as pool:
            parents_future = pool.submit(
                client.discover_parents,
                node_id=node_id,
                limit=20,
                api_key=api_key,
            )
            blanket_future = pool.submit(
                client.markov_blanket,
                node_id=node_id,
                limit=20,
                api_key=api_key,
            )

        parents = [_pick_node_id(item) for item in parents_future.result()]
        parents = [node for node in parents if node]
        blanket_nodes = [_pick_node_id(item) for item in blanket_future.result()]
        blanket_new = sorted(node for node in blanket_nodes if node and node not in parents)
        k_discovery = len(set(parents + blanket_new))
        return {
            "ticker": ticker,
            "source": "Abel CAP (live)",
            "parents": parents,
            "blanket_new": blanket_new,
            "children": [],
            "K_discovery": k_discovery,
            "note": f"K={k_discovery} tickers from Abel. Scan K = K x n_lags.",
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "source": f"abel_error: {exc}",
            "parents": [],
            "blanket_new": [],
            "children": [],
            "K_discovery": 0,
        }


def _resolve_workspace_api_key() -> str | None:
    token = resolve_api_key(env_path=".env")
    if token:
        return token

    skill_paths = (
        Path.home() / ".agents/skills/causal-abel/.env.skill",
        Path.home() / ".claude/skills/causal-abel/.env.skill",
    )
    for env_path in skill_paths:
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            if raw.startswith("ABEL_API_KEY="):
                return raw.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _pick_node_id(item: dict) -> str:
    for key in ("node_id", "id", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")
