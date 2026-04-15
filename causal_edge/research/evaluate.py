"""Immutable evaluation harness for research experiments.

This file is shipped by causal-edge. Agents do NOT write or modify it.
It enforces: validation gate, K auto-computation, result schema.

Usage:
    python -m causal_edge.research.evaluate [--workdir DIR]
    # or: causal-edge research --run
"""
from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from causal_edge.validation.gate import validate_strategy


def compute_k(strategy_path: Path) -> tuple[int, list[str], list[int]]:
    """Auto-compute K from strategy.py source via AST scan.

    Counts unique ticker string literals and lag integer values.
    K = n_tickers × n_unique_lags. Agent cannot self-report K.

    Returns (K, tickers_found, lags_found).
    """
    source = strategy_path.read_text()
    tree = ast.parse(source)

    tickers = set()
    lags = set()

    # Known crypto/equity suffixes
    ticker_pattern = re.compile(
        r'^[A-Z]{1,5}(USD)?$|^[A-Z]{2,5}-[A-Z]{1,2}$'
    )

    for node in ast.walk(tree):
        # String literals that look like tickers
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if ticker_pattern.match(val) and len(val) <= 10:
                tickers.add(val)

        # Integer literals in shift() calls — likely lags
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "shift":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                        if 1 <= arg.value <= 100:
                            lags.add(arg.value)
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                        if 1 <= kw.value.value <= 100:
                            lags.add(kw.value.value)

    # Also check for tuples/lists that define parent configs
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) >= 2:
            elts = node.elts
            if (isinstance(elts[0], ast.Constant) and isinstance(elts[0].value, str)
                    and ticker_pattern.match(elts[0].value)):
                tickers.add(elts[0].value)
                if isinstance(elts[1], ast.Constant) and isinstance(elts[1].value, int):
                    lags.add(elts[1].value)

    # Filter out common non-ticker strings
    non_tickers = {"SPY", "QQQ", "IWM", "TLT", "GLD"}  # market factors, not signal sources
    signal_tickers = tickers - non_tickers

    n_tickers = max(len(signal_tickers), 1)
    n_lags = max(len(lags), 1)
    K = n_tickers * n_lags

    return K, sorted(signal_tickers), sorted(lags)


def check_look_ahead(strategy_path: Path) -> list[str]:
    """Static look-ahead check on strategy.py source (T2-T5)."""
    from causal_edge.validation.look_ahead import check_static_file
    return check_static_file(strategy_path)


def run_evaluation(workdir: Path | str | None = None) -> dict:
    """Run strategy.py and validate via causal-edge.

    This is the ONLY way to produce a valid result. Agent cannot bypass this.

    Args:
        workdir: Path to research workspace. Default: current directory.

    Returns:
        Full validation result dict with verdict, score, K, metrics.
    """
    workdir = Path(workdir or ".")
    strategy_path = workdir / "strategy.py"

    if not strategy_path.exists():
        return _error("strategy.py not found. Run 'causal-edge research --init' first.")

    # 1. Static look-ahead check
    violations = check_look_ahead(strategy_path)
    if violations:
        return _error(f"Look-ahead violations: {violations}")

    # 2. Auto-compute K
    K, tickers, lags = compute_k(strategy_path)

    # 3. Import and run strategy
    import importlib.util
    spec = importlib.util.spec_from_file_location("strategy", str(strategy_path))
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)

    if not hasattr(strategy, "run_strategy"):
        return _error("strategy.py must define run_strategy() -> (pnl, dates, positions)")

    try:
        pnl, dates, positions = strategy.run_strategy()
        pnl = np.array(pnl, dtype=float)
        positions = np.array(positions, dtype=float)
    except Exception as e:
        return _error(f"strategy.run_strategy() failed: {e}")

    if len(pnl) < 30:
        return _error(f"Insufficient data: {len(pnl)} days (need 30+)")

    # 4. Export to temp CSV
    df = pd.DataFrame({"date": dates, "pnl": pnl, "position": positions})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name

    # 5. Validate via causal-edge (the GATE)
    try:
        result = validate_strategy(csv_path, profile="crypto_daily", K=K)
    except Exception as e:
        Path(csv_path).unlink(missing_ok=True)
        return _error(f"causal-edge validation failed: {e}")

    Path(csv_path).unlink(missing_ok=True)

    # 6. Enrich result with K info
    result["K"] = K
    result["K_detail"] = {"tickers": tickers, "lags": lags,
                          "n_tickers": len(tickers), "n_lags": len(lags)}

    return result


def append_results_tsv(workdir: Path, result: dict, status: str,
                       mode: str, description: str, commit: str = "none") -> None:
    """Append experiment result to results.tsv with schema validation.

    KEEP requires verdict == PASS. This is enforced here.
    """
    if status == "keep" and result.get("verdict") != "PASS":
        raise ValueError(
            f"Cannot KEEP with verdict={result.get('verdict')}. "
            f"KEEP requires verdict=PASS. Use status='discard'."
        )

    m = result.get("metrics", {})
    row = {
        "commit": commit,
        "lo_adj": round(m.get("lo_adjusted", 0), 3),
        "ic": round(m.get("ic", 0), 4),
        "omega": round(m.get("omega", 0), 3),
        "sharpe": round(m.get("sharpe", 0), 3),
        "pnl": round(m.get("total_pnl", 0) * 100, 1),
        "K": result.get("K", "?"),
        "score": result.get("score", "?/?"),
        "status": status,
        "mode": mode,
        "description": description,
    }

    tsv_path = workdir / "results.tsv"
    header = "commit\tlo_adj\tic\tomega\tsharpe\tpnl\tK\tscore\tstatus\tmode\tdescription\n"
    if not tsv_path.exists():
        tsv_path.write_text(header)

    line = "\t".join(str(row[k]) for k in
                     ["commit", "lo_adj", "ic", "omega", "sharpe", "pnl",
                      "K", "score", "status", "mode", "description"])
    with open(tsv_path, "a") as f:
        f.write(line + "\n")


def _error(msg: str) -> dict:
    return {"verdict": "ERROR", "score": "0/0", "failures": [msg],
            "metrics": {}, "triangle": {"ratio": 0, "rank": 0, "shape": 0},
            "K": 0}


def main():
    """CLI entry point: python -m causal_edge.research.evaluate"""
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate research strategy")
    parser.add_argument("--workdir", default=".", help="Research workspace dir")
    args = parser.parse_args()

    result = run_evaluation(args.workdir)

    # Print structured output
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("verdict") == "PASS" else 1)


if __name__ == "__main__":
    main()
