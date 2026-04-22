"""Evaluation helpers for engine-backed research strategies."""

from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from causal_edge.engine.backtest import run_backtest
from causal_edge.engine.feed_contract import FeedContractError
from causal_edge.engine.loader import load_engine_from_file
from causal_edge.engine.runtime_contract import DecisionContractError
from causal_edge.engine.signal_contract import SignalContractError
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


def run_preflight(
    workdir: Path | str | None = None,
    *,
    start: str | None = None,
    context_json: Path | None = None,
) -> dict:
    try:
        prepared = _prepare_engine_runtime(
            workdir=workdir,
            start=start,
            context_json=context_json,
        )
    except _PreparedRuntimeError as exc:
        return exc.payload

    semantic = _build_semantic_result(
        compiled=prepared["compiled"],
        engine=prepared["engine"],
        static_violations=prepared["static_violations"],
    )
    result = {
        "verdict": semantic["verdict"],
        "score": "semantic",
        "failures": list(semantic["failures"]),
        "warnings": list(semantic["warnings"]),
        "metrics": {},
        "triangle": {"ratio": 0, "rank": 0, "shape": 0},
        "K": prepared["k_value"],
        "requested_window": {"start": start, "end": None},
        "effective_window": _effective_window(
            pd.DataFrame({"date": prepared["compiled"].decision_index})
        ),
        "context_path": str(context_json.resolve()) if context_json is not None else None,
        "implementation_contract": prepared["compiled"].output_mode,
        "active_days": semantic["signal"]["active_days"],
        "total_days": semantic["signal"]["total_days"],
        "K_detail": {
            "tickers": prepared["tickers"],
            "lags": prepared["lags"],
            "n_tickers": len(prepared["tickers"]),
            "n_lags": len(prepared["lags"]),
        },
        "diagnostics": _build_preflight_diagnostics(semantic),
        "semantic": semantic,
    }
    _attach_semantic_artifacts(result, semantic=semantic, engine=prepared["engine"])
    return result


def run_evaluation(
    workdir: Path | str | None = None,
    *,
    start: str | None = None,
    context_json: Path | None = None,
    output_csv: Path | None = None,
) -> dict:
    try:
        prepared = _prepare_engine_runtime(
            workdir=workdir,
            start=start,
            context_json=context_json,
        )
    except _PreparedRuntimeError as exc:
        return exc.payload

    compiled = prepared["compiled"]
    engine = prepared["engine"]
    positions = compiled.positions
    dates = compiled.decision_index
    prices = compiled.close_prices
    semantic = _build_semantic_result(
        compiled=compiled,
        engine=engine,
        static_violations=prepared["static_violations"],
    )
    if semantic["verdict"] == "ERROR":
        result = _error(
            semantic["failures"][0] if semantic["failures"] else "Semantic preflight failed.",
            implementation_contract=compiled.output_mode,
            runtime_stage="semantic_preflight",
            signal=semantic["signal"],
        )
        result["warnings"] = list(semantic["warnings"])
        result["semantic"] = semantic
        _attach_semantic_artifacts(result, semantic=semantic, engine=engine)
        return result

    input_semantics = (
        "next_position" if compiled.output_mode == "decision_context" else "effective_position"
    )
    backtest = run_backtest(
        compiled.next_position if input_semantics == "next_position" else positions,
        prices,
        dates=dates,
        input_semantics=input_semantics,
        execution_delay_bars=compiled.runtime_profile.execution_delay_bars,
    )
    frame = _metric_input_frame(dates, backtest)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as handle:
        frame.to_csv(handle.name, index=False)
        csv_path = Path(handle.name)

    try:
        result = validate_strategy(csv_path, dsr_trials=prepared["k_value"])
    except Exception as exc:
        csv_path.unlink(missing_ok=True)
        return _error(
            f"causal-edge validation failed: {exc}",
            implementation_contract="engine",
            runtime_stage="validation",
        )

    csv_path.unlink(missing_ok=True)
    result["K"] = prepared["k_value"]
    result["requested_window"] = {"start": start, "end": None}
    result["effective_window"] = _effective_window(frame)
    result["context_path"] = str(context_json.resolve()) if context_json is not None else None
    result["implementation_contract"] = compiled.output_mode
    result["active_days"] = int((frame["position"].abs() > 0.01).sum())
    result["total_days"] = int(len(frame))
    result["K_detail"] = {
        "tickers": prepared["tickers"],
        "lags": prepared["lags"],
        "n_tickers": len(prepared["tickers"]),
        "n_lags": len(prepared["lags"]),
    }
    result["diagnostics"] = _build_runtime_diagnostics(result, frame)
    result["semantic"] = semantic
    _attach_semantic_artifacts(result, semantic=semantic, engine=engine)
    return result


def render_validation_markdown(result: dict) -> str:
    metrics = result.get("metrics", {})
    triangle = result.get("triangle", {})
    requested_window = result.get("requested_window", {})
    effective_window = result.get("effective_window", {})
    semantic = result.get("semantic") or {}
    prepared_inputs = semantic.get("prepared_inputs") or {}
    return f"""# Evaluation Summary

## Verdict

- verdict: `{result.get("verdict", "ERROR")}`
- score: `{result.get("score", "?/?")}`
- implementation_contract: `{result.get("implementation_contract", "unknown")}`
- K: `{result.get("K", "?")}`
- requested_start: `{requested_window.get("start", "none")}`
- effective_window: `{effective_window.get("start", "unknown")} -> {effective_window.get("end", "unknown")}`
- active_days: `{result.get("active_days", 0)} / {result.get("total_days", 0)}`

## Semantic

- semantic_verdict: `{semantic.get("verdict", "unknown")}`
- decision_count: `{semantic.get("decision_count", 0)}`
- read_count: `{semantic.get("read_count", 0)}`
- output_shape: `{(semantic.get("output_shape") or {}).get("label", "unknown")}`

### Semantic Warnings

{_format_failures(semantic.get("warnings", []))}

## Prepared Inputs

- selected_inputs: `{len(prepared_inputs.get("selected_inputs") or [])}`
- traced_inputs: `{', '.join(prepared_inputs.get('traced_inputs') or []) or 'none'}`
- prepared_effective_window: `{((prepared_inputs.get('effective_window') or {}).get('start') or 'unknown')} -> {((prepared_inputs.get('effective_window') or {}).get('end') or 'unknown')}`
- prepared_issues: `{', '.join(item.get('kind', 'unknown') for item in (prepared_inputs.get('issues') or [])) or 'none'}`

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

## Diagnostics

- failure_signature: `{(result.get("diagnostics") or {}).get("failure_signature", "unknown")}`
- runtime_stage: `{(result.get("diagnostics") or {}).get("runtime_stage", "unknown")}`
- signal_activity: `{((result.get("diagnostics") or {}).get("signal") or {}).get("active_days", 0)} / {((result.get("diagnostics") or {}).get("signal") or {}).get("total_days", 0)}`
- unique_positions: `{((result.get("diagnostics") or {}).get("signal") or {}).get("unique_position_count", 0)}`

## Hints

{_format_failures((result.get("diagnostics") or {}).get("hints", []))}

## Failures

{_format_failures(result.get("failures", []))}
"""


class _PreparedRuntimeError(Exception):
    def __init__(self, payload: dict) -> None:
        super().__init__(payload.get("failures", ["runtime preparation failed"])[0])
        self.payload = payload


def _prepare_engine_runtime(
    *,
    workdir: Path | str | None,
    start: str | None,
    context_json: Path | None,
) -> dict:
    workspace = Path(workdir or ".")
    engine_path = workspace / "engine.py"
    if not engine_path.exists():
        raise _PreparedRuntimeError(
            _error(
                "engine.py not found; research branches must define a module-owned StrategyEngine subclass.",
                implementation_contract="unknown",
                runtime_stage="load_engine",
            )
        )

    static_violations = check_look_ahead(engine_path)
    k_value, tickers, lags = compute_k(engine_path)
    try:
        research_context = _build_research_context(
            workspace=workspace,
            start=start,
            context_json=context_json,
        )
    except ValueError as exc:
        raise _PreparedRuntimeError(
            _error(
                str(exc),
                implementation_contract="engine",
                runtime_stage="context_build",
            )
        ) from exc

    try:
        engine_cls = load_engine_from_file(engine_path)
        engine = engine_cls(context=research_context)
        compiled = engine.compute_runtime_output(start=start)
    except (DecisionContractError, FeedContractError, SignalContractError, ImportError, TypeError, ValueError) as exc:
        raise _PreparedRuntimeError(
            _error(
                f"engine evaluation failed: {exc}",
                implementation_contract="engine",
                runtime_stage="compute_strategy",
            )
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive catch for user engines
        raise _PreparedRuntimeError(
            _error(
                f"engine evaluation failed: {exc}",
                implementation_contract="engine",
                runtime_stage="compute_strategy",
            )
        ) from exc

    return {
        "workspace": workspace,
        "engine_path": engine_path,
        "engine": engine,
        "compiled": compiled,
        "k_value": k_value,
        "tickers": tickers,
        "lags": lags,
        "static_violations": static_violations,
    }


def _build_semantic_result(*, compiled, engine, static_violations: list[str]) -> dict:
    signal = _signal_summary(compiled.positions)
    failures: list[str] = []
    warnings: list[str] = []
    if len(compiled.positions) < 30:
        failures.append(f"Insufficient data: {len(compiled.positions)} days (need 30+)")
    if static_violations:
        warnings.append(
            "Static look-ahead heuristics found suspicious patterns. Treat these as review hints, not a blocking verdict."
        )
        warnings.extend(static_violations[:5])
    if signal["active_days"] == 0:
        warnings.append("Signal is flat across the sampled runtime output.")
    elif signal["unique_position_count"] <= 1 or signal["position_switches"] == 0:
        warnings.append("Signal stayed at one position level during preflight.")
    prepared_feedback = _prepared_input_feedback(compiled=compiled, engine=engine)
    failures.extend(prepared_feedback["failures"])
    warnings.extend(prepared_feedback["warnings"])

    output_shape = "dynamic_signal"
    if signal["total_days"] == 0:
        output_shape = "empty_output"
    elif signal["active_days"] == 0:
        output_shape = "all_flat"
    elif signal["unique_position_count"] <= 1:
        output_shape = "constant_position"

    return {
        "verdict": "ERROR" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "decision_count": int(len(compiled.decision_index)),
        "read_count": int(len(engine.latest_decision_trace())),
        "signal": signal,
        "output_shape": {
            "label": output_shape,
            "unique_position_count": signal["unique_position_count"],
        },
        "prepared_inputs": prepared_feedback["summary"],
    }


def _attach_semantic_artifacts(result: dict, *, semantic: dict, engine) -> None:
    result["decision_trace"] = engine.latest_decision_trace()
    ctx = engine._last_decision_context
    result["decision_preview"] = ctx.preview(limit=5) if ctx is not None else []
    result["sample_points"] = ctx.sample_points(limit=3) if ctx is not None else []
    semantic["trace_excerpt"] = result["decision_trace"][:8]
    semantic["decision_preview"] = result["decision_preview"]
    semantic["sample_points"] = result["sample_points"]


def _build_preflight_diagnostics(semantic: dict) -> dict:
    signal = semantic.get("signal") or _signal_summary(np.array([], dtype=float))
    failure_signature = "semantic_ready"
    prepared = semantic.get("prepared_inputs") or {}
    issue_kinds = [str(item.get("kind") or "") for item in (prepared.get("issues") or [])]
    if semantic.get("verdict") == "ERROR":
        failure_signature = "semantic_preflight_failed"
    elif "effective_window_collapse" in issue_kinds:
        failure_signature = "effective_window_collapse"
    elif "stale_input_tail" in issue_kinds:
        failure_signature = "stale_input_tail"
    elif signal["active_days"] == 0:
        failure_signature = "signal_always_flat"
    elif signal["unique_position_count"] <= 1 or signal["position_switches"] == 0:
        failure_signature = "constant_position"
    return {
        "failure_signature": failure_signature,
        "runtime_stage": "semantic_preflight",
        "signal": signal,
        "hints": list(semantic.get("warnings", [])),
    }


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
    ticker = (
        research_context.get("ticker")
        or (research_context.get("discovery") or {}).get("ticker")
        or ((research_context.get("branch_spec") or {}).get("target"))
    )
    research_context["_runtime_profile"] = {
        "profile": "daily",
        "target": str(ticker or "").strip().upper() or None,
        "decision_event": "bar_close",
        "execution_delay_bars": 1,
        "return_basis": "close_to_close",
    }
    research_context["_execution_constraints"] = {
        "long_only": False,
    }
    feeds = research_context.get("_feeds")
    if not isinstance(feeds, dict):
        feeds = {}
    if "primary" not in feeds:
        feeds["primary"] = {
            "name": "primary",
            "kind": "bars",
            "adapter": "abel",
            "timeframe": "1d",
            "symbol": str(ticker or "").strip().upper() or None,
            "profile": "daily",
        }
    research_context["_feeds"] = feeds
    return research_context


def _metric_input_frame(dates, backtest: dict) -> pd.DataFrame:
    frame = pd.DataFrame(
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
    if backtest.get("next_position") is not None:
        frame["next_position"] = backtest["next_position"]
    if backtest.get("decision_time") is not None:
        frame["decision_time"] = backtest["decision_time"]
    if backtest.get("effective_time") is not None:
        frame["effective_time"] = backtest["effective_time"]
    return frame


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


def _coerce_timestamp(value) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text:
        return None
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _error(
    message: str,
    *,
    implementation_contract: str,
    runtime_stage: str,
    signal: dict | None = None,
) -> dict:
    diagnostics = _build_error_diagnostics(
        message=message,
        runtime_stage=runtime_stage,
        signal=signal,
    )
    return {
        "verdict": "ERROR",
        "score": "0/0",
        "failures": [message],
        "metrics": {},
        "triangle": {"ratio": 0, "rank": 0, "shape": 0},
        "K": 0,
        "implementation_contract": implementation_contract,
        "active_days": diagnostics["signal"]["active_days"],
        "total_days": diagnostics["signal"]["total_days"],
        "diagnostics": diagnostics,
    }


def _format_failures(failures: list[str]) -> str:
    if not failures:
        return "- none"
    return "\n".join(f"- {failure}" for failure in failures)


def _build_runtime_diagnostics(result: dict, frame: pd.DataFrame) -> dict:
    signal = _signal_summary(frame["position"].to_numpy(dtype=float))
    metrics = result.get("metrics") or {}
    semantic = result.get("semantic") or {}
    prepared = semantic.get("prepared_inputs") or {}
    issue_kinds = [str(item.get("kind") or "") for item in (prepared.get("issues") or [])]
    failure_signature = "healthy_signal"
    hints: list[str] = []
    if signal["total_days"] == 0:
        failure_signature = "no_usable_data"
        hints.append("No usable bars survived the requested evaluation window.")
    elif "effective_window_collapse" in issue_kinds:
        failure_signature = "effective_window_collapse"
        hints.append("Prepared inputs only become usable after the requested evaluation start.")
    elif "stale_input_tail" in issue_kinds:
        failure_signature = "stale_input_tail"
        hints.append("Prepared inputs stop before the evaluated decision tail, so late bars rely on stale auxiliary data.")
    elif signal["active_days"] == 0:
        failure_signature = "signal_always_flat"
        hints.append("Positions never left zero. Check data readiness and threshold logic.")
    elif signal["unique_position_count"] <= 1 or signal["position_switches"] == 0:
        failure_signature = "constant_position"
        hints.append("The engine produced only one position level. Check whether the signal ever changes sign.")
    elif result.get("verdict") != "PASS" and abs(float(metrics.get("position_ic", 0) or 0)) < 1e-12:
        failure_signature = "zero_information_signal"
        hints.append("position_ic stayed at 0. Inspect feature construction, alignment, and active-day coverage.")
    elif result.get("verdict") != "PASS":
        failure_signature = "validation_failed"
        hints.append("The engine executed, but validation metrics did not clear the research gate.")
    else:
        hints.append("Signal execution and validation both completed successfully.")
    return {
        "failure_signature": failure_signature,
        "runtime_stage": "validation",
        "signal": signal,
        "hints": hints + [item.get("message", "") for item in (prepared.get("issues") or []) if item.get("message")],
    }


def _prepared_input_feedback(*, compiled, engine) -> dict:
    context = engine.context or {}
    window = context.get("window_availability")
    data_manifest = context.get("data_manifest")
    trace = engine.latest_decision_trace()
    if not isinstance(window, dict):
        window = {}
    if not isinstance(data_manifest, dict):
        data_manifest = {}

    selected_inputs = [
        str(item.get("node_id") or "")
        for item in (data_manifest.get("selected_inputs") or [])
        if isinstance(item, dict) and str(item.get("node_id") or "").strip()
    ]
    traced_inputs = sorted(
        {
            str(item.get("feed") or "")
            for item in trace
            if str(item.get("feed") or "").strip()
            and str(item.get("feed")) not in {"primary", "target", str(compiled.runtime_profile.target or "")}
        }
    )
    warnings: list[str] = []
    failures: list[str] = []
    issues: list[dict[str, object]] = []

    effective_window = (window.get("effective_window") or {}) if isinstance(window, dict) else {}
    decision_start = compiled.decision_index[0] if len(compiled.decision_index) else None
    decision_end = compiled.decision_index[-1] if len(compiled.decision_index) else None
    effective_start = _coerce_timestamp(effective_window.get("start"))
    effective_end = _coerce_timestamp(effective_window.get("end"))
    if effective_start is not None and decision_start is not None and effective_start > decision_start:
        message = (
            "Prepared input availability starts after the first evaluated decision bar "
            f"({effective_start.date().isoformat()} > {pd.Timestamp(decision_start).date().isoformat()})."
        )
        warnings.append(message)
        issues.append({"kind": "effective_window_collapse", "severity": "warning", "message": message})
    if effective_end is not None and decision_end is not None and effective_end < decision_end:
        message = (
            "Prepared input availability ends before the last evaluated decision bar "
            f"({effective_end.date().isoformat()} < {pd.Timestamp(decision_end).date().isoformat()})."
        )
        warnings.append(message)
        issues.append({"kind": "stale_input_tail", "severity": "warning", "message": message})

    coverage = {
        str(item.get("node_id") or ""): item
        for item in (window.get("per_input_coverage") or [])
        if isinstance(item, dict) and str(item.get("node_id") or "").strip()
    }
    for node_id in traced_inputs:
        item = coverage.get(node_id)
        if not item:
            continue
        status = str(item.get("status") or "unknown")
        if status in {"error", "no_data", "no_target_overlap"}:
            message = f"Prepared input `{node_id}` was read at runtime but its prepared coverage status is `{status}`."
            failures.append(message)
            issues.append({"kind": "input_coverage_error", "severity": "error", "message": message})
        elif status in {"partial_target_overlap", "target_unavailable"}:
            message = f"Prepared input `{node_id}` only partially covers the target decision window."
            warnings.append(message)
            issues.append({"kind": "input_partial_coverage", "severity": "warning", "message": message})

    if selected_inputs and not traced_inputs:
        message = "Prepared auxiliary inputs were selected, but the strategy never read any non-primary prepared input."
        warnings.append(message)
        issues.append({"kind": "unused_prepared_inputs", "severity": "warning", "message": message})

    return {
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "selected_inputs": selected_inputs,
            "traced_inputs": traced_inputs,
            "effective_window": effective_window,
            "issues": issues,
        },
    }


def _build_error_diagnostics(
    *,
    message: str,
    runtime_stage: str,
    signal: dict | None = None,
) -> dict:
    failure_signature, hints = _classify_error_message(message)
    return {
        "failure_signature": failure_signature,
        "runtime_stage": runtime_stage,
        "signal": signal or _signal_summary(np.array([], dtype=float)),
        "hints": hints,
    }


def _classify_error_message(message: str) -> tuple[str, list[str]]:
    text = str(message or "").lower()
    if "must be utc-aware" in text or "must be normalized to midnight utc" in text:
        return (
            "datetime_contract_violation",
            [
                "The engine emitted dates outside the supported daily UTC runtime contract.",
                "For decision-context engines, build outputs with `ctx.decisions(...)`; for legacy engines, prefer `self.finalize_signals(...)`.",
            ],
        )
    if "aligned to strategy dates without gaps" in text or "unsupported alignment method" in text:
        return (
            "alignment_collapse",
            [
                "A required series could not be aligned safely to the strategy dates.",
                "Trim drivers to overlapping history or allow only explicitly justified gap handling.",
            ],
        )
    if "insufficient data: 0 days" in text or "no bars returned" in text:
        return (
            "no_usable_data",
            [
                "No usable market data was available for the requested window.",
                "Run `causal-edge verify-data` on the discovery payload before editing the branch further.",
            ],
        )
    if "insufficient data:" in text:
        return (
            "insufficient_history",
            [
                "The engine produced too little history for validation.",
                "Check the requested window and whether upstream filters removed most observations.",
            ],
        )
    if "look-ahead violations" in text:
        return (
            "look_ahead_violation",
            [
                "Static look-ahead checks found a forward-looking pattern in engine.py.",
            ],
        )
    if "not available inside compute_decisions" in text:
        return (
            "decision_context_escape_hatch",
            [
                "The branch tried to bypass DecisionContext with a raw data helper.",
                "Read target and driver data through ctx.target / ctx.feed / ctx.points instead.",
            ],
        )
    if "api key" in text or "oauth" in text:
        return (
            "auth_missing",
            [
                "Abel auth was missing for a data fetch path.",
                "Run `causal-edge login` or provide a workspace `.env` with ABEL_API_KEY.",
            ],
        )
    if "causal-edge validation failed:" in text:
        return (
            "validation_failed",
            [
                "Signal generation completed, but validation could not finish cleanly.",
                "Re-run `causal-edge debug-evaluate --workdir ...` and inspect the validation stage diagnostics.",
            ],
        )
    return (
        "engine_runtime_error",
        [
            "The engine failed before validation could score it.",
            "Use `causal-edge debug-evaluate --workdir ...` to inspect the runtime diagnostics.",
        ],
    )


def _signal_summary(positions) -> dict:
    arr = np.asarray(positions, dtype=float)
    if arr.size == 0:
        return {
            "active_days": 0,
            "total_days": 0,
            "unique_position_count": 0,
            "unique_positions": [],
            "position_switches": 0,
        }
    rounded = np.round(arr, 8)
    unique_positions = sorted({float(value) for value in rounded.tolist()})
    switches = int(np.count_nonzero(np.abs(np.diff(rounded)) > 1e-8)) if len(rounded) > 1 else 0
    return {
        "active_days": int((np.abs(arr) > 0.01).sum()),
        "total_days": int(len(arr)),
        "unique_position_count": len(unique_positions),
        "unique_positions": unique_positions[:12],
        "position_switches": switches,
    }


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
