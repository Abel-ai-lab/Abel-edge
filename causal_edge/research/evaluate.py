"""Immutable evaluation harness for research experiments."""

from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from causal_edge.research.constants import RESULTS_COLUMNS, RESULTS_HEADER
from causal_edge.research.workspace import read_results_rows
from causal_edge.validation.gate import validate_strategy
from causal_edge.validation.gate_logic import decide_keep_discard
from causal_edge.validation.metrics import load_profile

NON_TICKERS = {"SPY", "QQQ", "IWM", "TLT", "GLD"}
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(USD)?$|^[A-Z]{2,5}-[A-Z]{1,2}$")


def compute_k(strategy_path: Path) -> tuple[int, list[str], list[int]]:
    """Auto-compute K from strategy.py AST: unique tickers x unique lags."""
    source = strategy_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    tickers: set[str] = set()
    lags: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.strip()
            if TICKER_PATTERN.match(value) and len(value) <= 10:
                tickers.add(value)
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

    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) >= 2:
            first, second = node.elts[0], node.elts[1]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                if TICKER_PATTERN.match(first.value):
                    tickers.add(first.value)
                    if isinstance(second, ast.Constant) and isinstance(second.value, int):
                        lags.add(second.value)

    signal_tickers = sorted(tickers - NON_TICKERS)
    lag_values = sorted(lags)
    k_value = max(len(signal_tickers), 1) * max(len(lag_values), 1)
    return k_value, signal_tickers, lag_values


def check_look_ahead(strategy_path: Path) -> list[str]:
    from causal_edge.validation.look_ahead import check_static_file

    return check_static_file(strategy_path)


def run_evaluation(workdir: Path | str | None = None) -> dict:
    workspace = Path(workdir or ".")
    strategy_path = workspace / "strategy.py"
    if not strategy_path.exists():
        return _error("strategy.py not found. Run 'causal-edge research init' first.")

    violations = check_look_ahead(strategy_path)
    if violations:
        return _error(f"Look-ahead violations: {violations}")

    k_value, tickers, lags = compute_k(strategy_path)
    strategy_module = _load_strategy_module(strategy_path)
    if not hasattr(strategy_module, "run_strategy"):
        return _error("strategy.py must define run_strategy() -> (pnl, dates, positions)")

    try:
        pnl, dates, positions = strategy_module.run_strategy()
        pnl = np.asarray(pnl, dtype=float)
        positions = np.asarray(positions, dtype=float)
    except Exception as exc:
        return _error(f"strategy.run_strategy() failed: {exc}")

    if len(pnl) < 30:
        return _error(f"Insufficient data: {len(pnl)} days (need 30+)")

    frame = pd.DataFrame({"date": dates, "pnl": pnl, "position": positions})
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as handle:
        frame.to_csv(handle.name, index=False)
        csv_path = Path(handle.name)

    try:
        result = validate_strategy(csv_path, profile="crypto_daily", dsr_trials=k_value)
    except Exception as exc:
        csv_path.unlink(missing_ok=True)
        return _error(f"causal-edge validation failed: {exc}")

    csv_path.unlink(missing_ok=True)
    result["K"] = k_value
    result["K_detail"] = {
        "tickers": tickers,
        "lags": lags,
        "n_tickers": len(tickers),
        "n_lags": len(lags),
    }
    return result


def append_results_tsv(
    workdir: Path,
    result: dict,
    status: str,
    mode: str,
    description: str,
    *,
    exp_id: str,
    ticker: str,
    branch_id: str,
    round_id: str,
    decision: str,
    validation_path: str,
    commit: str = "none",
) -> None:
    if status == "keep" and result.get("verdict") != "PASS":
        raise ValueError(
            f"Cannot KEEP with verdict={result.get('verdict')}. "
            "KEEP requires verdict=PASS. Use status='discard'."
        )

    metrics = result.get("metrics", {})
    row = {
        "exp_id": exp_id,
        "ticker": ticker,
        "branch_id": branch_id,
        "round_id": round_id,
        "decision": decision,
        "commit": commit,
        "lo_adj": round(metrics.get("lo_adjusted", 0), 3),
        "ic": round(metrics.get("position_ic", 0), 4),
        "omega": round(metrics.get("omega", 0), 3),
        "sharpe": round(metrics.get("sharpe", 0), 3),
        "max_dd": round(metrics.get("max_dd", 0), 4),
        "pnl": round(metrics.get("total_return", 0) * 100, 1),
        "K": result.get("K", "?"),
        "score": result.get("score", "?/?"),
        "verdict": result.get("verdict", "ERROR"),
        "status": status,
        "mode": mode,
        "description": description,
        "validation_path": validation_path,
    }

    tsv_path = Path(workdir) / "results.tsv"
    if not tsv_path.exists():
        tsv_path.write_text(RESULTS_HEADER, encoding="utf-8")

    line = "\t".join(str(row[key]) for key in RESULTS_COLUMNS)
    with tsv_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def decide_research_outcome(workdir: Path, result: dict) -> tuple[str, str]:
    if result.get("verdict") != "PASS":
        return "discard", "discard"

    rows = read_results_rows(workdir)
    keep_rows = [row for row in rows if row.get("status") == "keep"]
    if not keep_rows:
        return "keep", "keep"

    baseline = keep_rows[-1]
    baseline_metrics = {
        "lo_adjusted": float(baseline.get("lo_adj") or 0),
        "position_ic": float(baseline.get("ic") or 0),
        "omega": float(baseline.get("omega") or 0),
        "sharpe": float(baseline.get("sharpe") or 0),
        "max_dd": float(baseline.get("max_dd") or 0),
        "total_return": float(baseline.get("pnl") or 0) / 100.0,
    }
    current_metrics = result.get("metrics", {})
    profile = load_profile("crypto_daily")
    decision = decide_keep_discard(current_metrics, baseline_metrics, profile).lower()
    status = "keep" if decision == "keep" else "discard"
    return status, decision


def _load_strategy_module(strategy_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("research_strategy", str(strategy_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _error(message: str) -> dict:
    return {
        "verdict": "ERROR",
        "score": "0/0",
        "failures": [message],
        "metrics": {},
        "triangle": {"ratio": 0, "rank": 0, "shape": 0},
        "K": 0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate research strategy")
    parser.add_argument("--workdir", default=".", help="Research workspace dir")
    args = parser.parse_args()

    result = run_evaluation(args.workdir)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("verdict") == "PASS" else 1)


if __name__ == "__main__":
    main()
