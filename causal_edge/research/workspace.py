"""Research workspace initialization.

Creates a workspace with:
  strategy.py  — empty shell for agent to fill
  results.tsv  — empty with header
  memory.md    — empty template
  discovery.json — Abel parents (if API key available)

evaluate.py is NOT copied — it runs from causal_edge.research.evaluate.
This prevents agents from modifying it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


STRATEGY_TEMPLATE = '''"""Strategy for {ticker} — experiment baseline.

Fill in run_strategy(). Everything else is handled by causal-edge.
Run: python -m causal_edge.research.evaluate --workdir .
"""
import numpy as np
import pandas as pd


def run_strategy():
    """Your strategy logic. Returns (pnl, dates, positions).

    pnl: np.ndarray of daily log-return PnL
    dates: pd.DatetimeIndex
    positions: np.ndarray of daily position sizes (0=flat, 1=long)
    """
    # TODO: implement your strategy
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

RESULTS_HEADER = "commit\tlo_adj\tic\tomega\tsharpe\tpnl\tK\tscore\tstatus\tmode\tdescription\n"


def init_workspace(ticker: str, workdir: Path | str | None = None) -> Path:
    """Create research workspace for a ticker.

    Args:
        ticker: Target asset ticker (e.g., "SOLUSD", "TSLA")
        workdir: Where to create. Default: ./research/<ticker>/

    Returns:
        Path to created workspace.
    """
    ticker = ticker.upper()
    if workdir is None:
        workdir = Path("research") / ticker.lower()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # strategy.py — empty shell
    strategy_path = workdir / "strategy.py"
    if not strategy_path.exists():
        strategy_path.write_text(STRATEGY_TEMPLATE.format(ticker=ticker))

    # results.tsv — empty with header
    results_path = workdir / "results.tsv"
    if not results_path.exists():
        results_path.write_text(RESULTS_HEADER)

    # memory.md — empty template
    memory_path = workdir / "memory.md"
    if not memory_path.exists():
        memory_path.write_text(MEMORY_TEMPLATE.format(ticker=ticker))

    # discovery.json — try Abel, fallback to template
    discovery_path = workdir / "discovery.json"
    if not discovery_path.exists():
        discovery = _try_abel_discovery(ticker)
        discovery_path.write_text(json.dumps(discovery, indent=2))

    return workdir


def _try_abel_discovery(ticker: str) -> dict:
    """Try to run Abel discovery. Returns discovery dict."""
    api_key = os.environ.get("ABEL_API_KEY", "")

    # Check shared skill env files
    if not api_key:
        for env_path in [
            Path.home() / ".agents/skills/causal-abel/.env.skill",
            Path.home() / ".claude/skills/causal-abel/.env.skill",
        ]:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("ABEL_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"')
                        break
            if api_key:
                break

    if not api_key:
        return {
            "ticker": ticker,
            "source": "template (no ABEL_API_KEY)",
            "parents": [],
            "blanket": [],
            "children": [],
            "K_discovery": 0,
            "note": "Set ABEL_API_KEY or run Abel discovery manually.",
        }

    # Run Abel discovery
    try:
        return _run_abel_discovery(ticker, api_key)
    except Exception as e:
        return {
            "ticker": ticker,
            "source": f"abel_error: {e}",
            "parents": [],
            "blanket": [],
            "children": [],
            "K_discovery": 0,
        }


def _run_abel_discovery(ticker: str, api_key: str) -> dict:
    """Query Abel CAP for parents + blanket + children (parallel)."""
    import urllib.request
    import urllib.error
    from concurrent.futures import ThreadPoolExecutor, as_completed

    base = "https://cap.abel.ai/api"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def _cap_call(verb: str, params: dict) -> dict:
        import uuid
        payload = json.dumps({
            "cap_version": "0.2.2",
            "request_id": str(uuid.uuid4()),
            "verb": verb,
            "params": params,
        }).encode()
        req = urllib.request.Request(f"{base}/cap", data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    # Normalize ticker
    node_id = ticker.upper()
    if "." not in node_id:
        if node_id.endswith("USD") and len(node_id) > 4:
            node_id = f"{node_id}.price"
        else:
            node_id = f"{node_id}.price"

    # Parallel: parents + blanket + children (3 independent queries)
    results = {}
    def _query_parents():
        r = _cap_call("graph.neighbors", {"node_id": node_id, "scope": "parents", "max_neighbors": 20})
        return [n["node_id"] for n in r.get("result", {}).get("neighbors", [])]

    def _query_blanket():
        try:
            r = _cap_call("extensions.abel.markov_blanket", {"target_node": node_id})
            return r.get("result", {}).get("markov_blanket", [])
        except Exception:
            return []

    def _query_children():
        r = _cap_call("graph.neighbors", {"node_id": node_id, "scope": "children", "max_neighbors": 20})
        return [n["node_id"] for n in r.get("result", {}).get("neighbors", [])]

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_parents = pool.submit(_query_parents)
        f_blanket = pool.submit(_query_blanket)
        f_children = pool.submit(_query_children)

    parents = f_parents.result()
    blanket = f_blanket.result()
    children = f_children.result()
    blanket_new = [b for b in blanket if b not in parents and b not in children]
    K = len(set(parents + blanket_new))

    return {
        "ticker": ticker,
        "source": "Abel CAP (live, parallel)",
        "parents": parents,
        "blanket_new": blanket_new,
        "children": children,
        "K_discovery": K,
        "note": f"K={K} tickers from Abel. Scan K = K × n_lags.",
    }
