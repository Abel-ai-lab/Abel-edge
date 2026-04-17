"""Evaluation helpers for engine-backed research strategies."""

from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

from causal_edge.engine.backtest import run_backtest
from causal_edge.engine.feed_contract import FeedContractError
from causal_edge.engine.loader import load_engine_from_file
from causal_edge.engine.signal_contract import SignalContractError, validate_signal_output
from causal_edge.research.handoff import build_strategy_handoff, write_strategy_handoff
from causal_edge.validation.gate import validate_strategy

NON_TICKERS = {"SPY", "QQQ", "IWM", "TLT", "GLD", "UTC", "D", "B"}
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}(USD)?$|^[A-Z0-9]{2,10}$|^[A-Z]{2,5}-[A-Z]{1,2}$")


def compute_k(source_path: Path) -> tuple[int, list[str], list[int]]:
    """Auto-compute K from engine.py AST: unique tickers x unique lags."""
    source = source_path.read_text(encoding="utf-8")
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


def check_look_ahead(source_path: Path) -> list[str]:
    from causal_edge.validation.look_ahead import check_static_file

    return check_static_file(source_path)


def run_evaluation(
    workdir: Path | str | None = None,
    *,
    start: str | None = None,
    context_json: Path | None = None,
    output_csv: Path | None = None,
) -> dict:
    workspace = Path(workdir or ".")
    engine_path = workspace / "engine.py"
    if not engine_path.exists():
        return _error(
            "engine.py not found; research branches must define a module-owned StrategyEngine subclass.",
            implementation_contract="unknown",
        )

    violations = check_look_ahead(engine_path)
    if violations:
        return _error(
            f"Look-ahead violations: {violations}",
            implementation_contract="engine",
        )

    k_value, tickers, lags = compute_k(engine_path)
    try:
        research_context = _build_research_context(
            workspace=workspace,
            start=start,
            context_json=context_json,
        )
    except ValueError as exc:
        return _error(
            str(exc),
            implementation_contract="engine",
        )

    try:
        engine_cls = load_engine_from_file(engine_path)
        engine = engine_cls(context=research_context)
        positions, dates, prices = validate_signal_output(
            *engine.compute_signals(),
            profile="daily",
        )
    except (FeedContractError, SignalContractError, ImportError, ValueError) as exc:
        return _error(
            f"engine evaluation failed: {exc}",
            implementation_contract="engine",
        )
    except Exception as exc:  # pragma: no cover - defensive catch for user engines
        return _error(
            f"engine evaluation failed: {exc}",
            implementation_contract="engine",
        )

    if len(positions) < 30:
        return _error(
            f"Insufficient data: {len(positions)} days (need 30+)",
            implementation_contract="engine",
        )

    backtest = run_backtest(positions, prices)
    frame = _metric_input_frame(dates, backtest)
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
        return _error(
            f"causal-edge validation failed: {exc}",
            implementation_contract="engine",
        )

    csv_path.unlink(missing_ok=True)
    result["K"] = k_value
    result["requested_window"] = {"start": start, "end": None}
    result["effective_window"] = _effective_window(frame)
    result["context_path"] = str(context_json.resolve()) if context_json is not None else None
    result["implementation_contract"] = "engine"
    result["active_days"] = int((frame["position"].abs() > 0.01).sum())
    result["total_days"] = int(len(frame))
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
- implementation_contract: `{result.get("implementation_contract", "unknown")}`
- K: `{result.get("K", "?")}`
- requested_start: `{requested_window.get("start", "none")}`
- effective_window: `{effective_window.get("start", "unknown")} -> {effective_window.get("end", "unknown")}`
- active_days: `{result.get("active_days", 0)} / {result.get("total_days", 0)}`

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
            strategy_path=workdir / "engine.py",
            result_path=json_path,
            report_path=markdown_path,
            handoff_path=handoff_path,
        )
        write_strategy_handoff(payload, handoff_path)


def _build_research_context(
    *,
    workspace: Path,
    start: str | None,
    context_json: Path | None,
) -> dict:
    injected: dict[str, object] = {}
    if context_json is not None:
        try:
            payload = json.loads(context_json.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid context JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Invalid context JSON: expected an object payload.")
        injected = dict(payload)

    research_context = dict(injected)
    research_context["_research"] = {
        "workdir": str(workspace.resolve()),
        "requested_window": {"start": start, "end": None},
    }
    data_contract = research_context.get("_data_contract")
    if not isinstance(data_contract, dict):
        data_contract = {}
    data_contract["profile"] = "daily"
    research_context["_data_contract"] = data_contract
    return research_context


def _metric_input_frame(dates, backtest: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "pnl": backtest["pnl"],
            "position": backtest["positions"],
            "asset_return": backtest["asset_returns"],
            "gross_pnl": backtest["gross_pnl"],
            "turnover": backtest["turnover"],
            "execution_cost": backtest["execution_cost"],
        }
    )


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


def _error(message: str, *, implementation_contract: str) -> dict:
    return {
        "verdict": "ERROR",
        "score": "0/0",
        "failures": [message],
        "metrics": {},
        "triangle": {"ratio": 0, "rank": 0, "shape": 0},
        "K": 0,
        "implementation_contract": implementation_contract,
        "active_days": 0,
        "total_days": 0,
    }


def _format_failures(failures: list[str]) -> str:
    if not failures:
        return "- none"
    return "\n".join(f"- {failure}" for failure in failures)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate engine-backed research strategy")
    parser.add_argument("--workdir", default=".", help="Directory containing engine.py")
    parser.add_argument("--start", default=None, help="Optional backtest start date")
    parser.add_argument("--context-json", default=None, help="Optional JSON context payload")
    parser.add_argument("--output-json", default=None, help="Optional path for raw JSON result")
    parser.add_argument("--output-md", default=None, help="Optional path for raw markdown report")
    parser.add_argument("--output-csv", default=None, help="Optional path for metric input CSV")
    parser.add_argument("--output-handoff", default=None, help="Optional path for handoff JSON")
    args = parser.parse_args()

    if args.output_handoff and (not args.output_json or not args.output_md):
        raise SystemExit("--output-handoff requires both --output-json and --output-md.")

    result = run_evaluation(
        args.workdir,
        start=args.start,
        context_json=Path(args.context_json) if args.context_json else None,
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
