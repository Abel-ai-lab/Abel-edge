"""Raw evaluation helpers for experimental strategies."""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from causal_edge.research.handoff import build_strategy_handoff, write_strategy_handoff
from causal_edge.validation.gate import validate_strategy

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


def run_evaluation(
    workdir: Path | str | None = None,
    *,
    start: str | None = None,
    output_csv: Path | None = None,
) -> dict:
    workspace = Path(workdir or ".")
    strategy_path = workspace / "strategy.py"
    if not strategy_path.exists():
        return _error("strategy.py not found")

    violations = check_look_ahead(strategy_path)
    if violations:
        return _error(f"Look-ahead violations: {violations}")

    k_value, tickers, lags = compute_k(strategy_path)
    strategy_module = _load_strategy_module(strategy_path)
    if not hasattr(strategy_module, "run_strategy"):
        return _error("strategy.py must define run_strategy() -> (pnl, dates, positions)")

    try:
        pnl, dates, positions = _run_strategy(strategy_module.run_strategy, start=start)
        pnl = np.asarray(pnl, dtype=float)
        positions = np.asarray(positions, dtype=float)
    except Exception as exc:
        return _error(f"strategy.run_strategy() failed: {exc}")

    if len(pnl) < 30:
        return _error(f"Insufficient data: {len(pnl)} days (need 30+)")

    frame = pd.DataFrame({"date": dates, "pnl": pnl, "position": positions})
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as handle:
        frame.to_csv(handle.name, index=False)
        csv_path = Path(handle.name)

    try:
        result = validate_strategy(csv_path, dsr_trials=k_value)
    except Exception as exc:
        csv_path.unlink(missing_ok=True)
        return _error(f"causal-edge validation failed: {exc}")

    csv_path.unlink(missing_ok=True)
    result["K"] = k_value
    result["requested_window"] = {"start": start, "end": None}
    result["effective_window"] = _effective_window(frame)
    result["K_detail"] = {
        "tickers": tickers,
        "lags": lags,
        "n_tickers": len(tickers),
        "n_lags": len(lags),
    }
    return result


def render_validation_markdown(result: dict) -> str:
    metrics = result.get("metrics", {})
    triangle = result.get("triangle", {})
    requested_window = result.get("requested_window", {})
    effective_window = result.get("effective_window", {})
    return f"""# Evaluation Summary

## Verdict

- verdict: `{result.get("verdict", "ERROR")}`
- score: `{result.get("score", "?/?")}`
- K: `{result.get("K", "?")}`
- requested_start: `{requested_window.get("start", "none")}`
- effective_window: `{effective_window.get("start", "unknown")} -> {effective_window.get("end", "unknown")}`

## Triangle

- lo_ratio: `{triangle.get("ratio", 0):.3f}`
- rank_ic: `{triangle.get("rank", 0):.4f}`
- omega_shape: `{triangle.get("shape", 0):.3f}`

## Metrics

- lo_adjusted: `{metrics.get("lo_adjusted", 0):.3f}`
- position_ic: `{metrics.get("position_ic", 0):.4f}`
- omega: `{metrics.get("omega", 0):.3f}`
- sharpe: `{metrics.get("sharpe", 0):.3f}`
- total_return: `{metrics.get("total_return", 0) * 100:.1f}%`
- max_dd: `{metrics.get("max_dd", 0) * 100:.1f}%`

## Failures

{_format_failures(result.get("failures", []))}
"""


def write_evaluation_outputs(
    result: dict,
    *,
    workdir: Path | None = None,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    handoff_path: Path | None = None,
) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_validation_markdown(result), encoding="utf-8")
    if handoff_path is not None:
        if workdir is None:
            raise ValueError("workdir is required when writing a strategy handoff.")
        if json_path is None or markdown_path is None:
            raise ValueError(
                "json_path and markdown_path are required when writing a strategy handoff."
            )
        payload = build_strategy_handoff(
            result,
            strategy_path=workdir / "strategy.py",
            result_path=json_path,
            report_path=markdown_path,
            handoff_path=handoff_path,
        )
        write_strategy_handoff(payload, handoff_path)


def _load_strategy_module(strategy_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("raw_eval_strategy", str(strategy_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_strategy(run_strategy, *, start: str | None):
    signature = inspect.signature(run_strategy)
    if "start" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return run_strategy(start=start)
    return run_strategy()


def _effective_window(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"start": None, "end": None}
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce").dropna()
    if dates.empty:
        return {"start": None, "end": None}
    return {
        "start": dates.min().date().isoformat(),
        "end": dates.max().date().isoformat(),
    }


def _error(message: str) -> dict:
    return {
        "verdict": "ERROR",
        "score": "0/0",
        "failures": [message],
        "metrics": {},
        "triangle": {"ratio": 0, "rank": 0, "shape": 0},
        "K": 0,
    }


def _format_failures(failures: list[str]) -> str:
    if not failures:
        return "- none"
    return "\n".join(f"- {failure}" for failure in failures)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a strategy and emit raw validation facts"
    )
    parser.add_argument("--workdir", default=".", help="Directory containing strategy.py")
    parser.add_argument(
        "--start", default=None, help="Optional backtest start date passed to run_strategy"
    )
    parser.add_argument("--output-json", default=None, help="Optional path for raw JSON result")
    parser.add_argument(
        "--output-md", default=None, help="Optional path for raw validation markdown"
    )
    parser.add_argument("--output-csv", default=None, help="Optional path for metric input CSV")
    parser.add_argument(
        "--output-handoff", default=None, help="Optional path for edge-owned handoff JSON"
    )
    args = parser.parse_args()
    if args.output_handoff and (not args.output_json or not args.output_md):
        raise SystemExit("--output-handoff requires both --output-json and --output-md.")

    result = run_evaluation(
        args.workdir,
        start=args.start,
        output_csv=Path(args.output_csv) if args.output_csv else None,
    )
    write_evaluation_outputs(
        result,
        workdir=Path(args.workdir),
        json_path=Path(args.output_json) if args.output_json else None,
        markdown_path=Path(args.output_md) if args.output_md else None,
        handoff_path=Path(args.output_handoff) if args.output_handoff else None,
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result.get("verdict") == "PASS" else 1)


if __name__ == "__main__":
    main()
