"""Strategy admission gate — validate before adding to production.

Usage:
    from abel_edge.validation import validate_strategy
    result = validate_strategy("strategies/my_strategy/engine.py",
                               trade_log="data/trade_log.csv")
    print(result["verdict"])   # PASS / FAIL
    print(result["failures"])  # list of failure messages
    print(result["triangle"])  # {lo, ic, omega} scores
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from abel_edge.validation.gate_explain import explain_metric_gates
from abel_edge.validation.metrics import (
    compute_all_metrics,
    detect_profile,
    load_profile,
)


def validate_strategy(
    trade_log: str | Path,
    profile: str | None = None,
    positions_col: str = "position",
    dsr_trials: int | None = None,
) -> dict:
    """Run full Abel Proof validation on a strategy's trade log.

    Args:
        trade_log: Path to trade_log CSV (must have 'date', 'pnl' columns).
        profile: Profile name ('crypto_daily', 'equity_daily', 'hft')
                 or path to YAML. Auto-detected if None.
        positions_col: Column name for positions (default 'position').
        dsr_trials: Optional externally declared strategy exploration count used
                    by DSR. Falls back to the profile default when omitted.

    Returns dict with:
        verdict: "PASS" or "FAIL"
        score: "N/M" (e.g. "4/5")
        failures: list of failure message strings
        metrics: full metrics dict
        triangle: {ratio, rank, shape} — the three leverage-invariant dims
        profile: profile name used
    """
    df = pd.read_csv(trade_log, parse_dates=["date"])
    if "source" in df.columns:
        source = df["source"].astype(str).str.lower()
        backtest_df = df[source != "live"].copy()
        if len(backtest_df) > 0:
            df = backtest_df
    if len(df) < 30:
        return {
            "verdict": "FAIL",
            "score": "0/0",
            "failures": [f"Insufficient data: {len(df)} rows (need 30+)"],
            "warnings": [],
            "metrics": {},
            "triangle": {"ratio": 0, "rank": 0, "shape": 0},
            "profile": "unknown",
        }

    pnl = df["pnl"].values.astype(float)
    dates = pd.DatetimeIndex(df["date"])
    positions = df[positions_col].values.astype(float) if positions_col in df.columns else None
    asset_returns = (
        df["asset_return"].values.astype(float) if "asset_return" in df.columns else None
    )

    execution_cost = (
        df["execution_cost"].values.astype(float) if "execution_cost" in df.columns else None
    )

    # PnL consistency check: |pnl - position * asset_return - execution_cost| should be small
    warnings: list[str] = []
    if positions is not None and asset_returns is not None:
        expected_pnl = positions * asset_returns
        if execution_cost is not None:
            expected_pnl = expected_pnl - execution_cost
        residuals = np.abs(pnl - expected_pnl)
        p95 = float(np.percentile(residuals, 95))
        if p95 >= 0.01:
            basis = "position * asset_return - execution_cost"
            if execution_cost is None:
                basis = "position * asset_return"
            warnings.append(
                f"PnL consistency: 95th percentile of |pnl - {basis}| = {p95:.4f} "
                f"(threshold 0.01). Check for fees, slippage, or data errors."
            )

    # Auto-detect or load profile
    if profile is None:
        profile_name = detect_profile(pnl, dates, asset_returns=asset_returns)
    else:
        profile_name = profile
    prof = load_profile(profile_name)
    validation_cfg = prof.get("validation", {})

    # Runtime look-ahead check. Prefer explicit asset returns when available and
    # fall back to pnl / position on active days so the check remains live.
    from abel_edge.validation.look_ahead import check_runtime

    pos_for_check = positions if positions is not None else np.zeros(len(pnl), dtype=float)
    if asset_returns is not None and len(asset_returns) == len(pnl):
        returns_for_check = asset_returns
    else:
        returns_for_check = np.divide(
            pnl,
            pos_for_check,
            out=np.zeros_like(pnl, dtype=float),
            where=np.abs(pos_for_check) > 0.01,
        )
    la_messages = check_runtime(
        pnl,
        pos_for_check,
        returns_for_check,
        threshold=float(validation_cfg.get("look_ahead_mag_corr_max", 0.3)),
        hit_rate_max=float(validation_cfg.get("look_ahead_hit_rate_max", 0.70)),
    )

    # Compute all metrics
    if positions is not None:
        metrics = compute_all_metrics(
            pnl,
            dates,
            positions,
            prof,
            dsr_trials=dsr_trials,
            asset_returns=asset_returns,
        )
    else:
        metrics = compute_all_metrics(
            pnl,
            dates,
            profile=prof,
            dsr_trials=dsr_trials,
            asset_returns=asset_returns,
        )

    gate_explanation = explain_metric_gates(metrics, prof)
    passed = bool(gate_explanation["passed"])
    failures = list(gate_explanation["failures"])
    score = str(gate_explanation["score"])

    # Extract triangle
    mt = prof.get("metric_triangle", {})
    opt_key = {"lo_adjusted_sharpe": "lo_adjusted", "sharpe": "sharpe", "total_return": "total_return"}.get(
        mt.get("optimize", "lo_adjusted_sharpe"), "lo_adjusted"
    )
    triangle = {
        "ratio": metrics.get(opt_key, 0),
        "rank": metrics.get("position_ic", 0),
        "shape": metrics.get("omega", 0),
    }

    runtime_failures = [message for message in la_messages if message.startswith("R1")]
    runtime_warnings = [message for message in la_messages if not message.startswith("R1")]
    if runtime_failures:
        failures.extend(runtime_failures)
        passed = False
    warnings.extend(runtime_warnings)

    return {
        "verdict": "PASS" if passed else "FAIL",
        "score": score,
        "failures": failures,
        "warnings": warnings,
        "metrics": metrics,
        "triangle": triangle,
        "profile": profile_name,
    }


def validate_all_strategies(config_path: str | Path | None = None) -> dict:
    """Validate all strategies in strategies.yaml.

    Returns dict mapping strategy_id → validation result.
    """
    from abel_edge.config import load_config

    cfg = load_config(config_path)
    results = {}
    for s_cfg in cfg["strategies"]:
        sid = s_cfg["id"]
        log_path = s_cfg.get("trade_log", "")
        if not Path(log_path).exists():
            results[sid] = {
                "verdict": "SKIP",
                "score": "0/0",
                "failures": [f"Trade log not found: {log_path}"],
                "warnings": [],
                "metrics": {},
                "triangle": {"ratio": 0, "rank": 0, "shape": 0},
                "profile": "unknown",
            }
            continue
        results[sid] = validate_strategy(log_path)
    return results


def print_validation_report(results: dict) -> None:
    """Print a formatted validation report."""
    print("=" * 70)
    print("ABEL PROOF VALIDATION REPORT")
    print("=" * 70)

    for sid, r in results.items():
        tri = r["triangle"]
        badge = r["score"]
        verdict = r["verdict"]
        if verdict == "PASS":
            status, marker = "PASS", "+"
        elif verdict == "SKIP":
            status, marker = "SKIP", "-"
        else:
            status, marker = "FAIL", "x"
        print(f"\n  [{marker}] {sid:15s}  {badge:>6s}  {status}")
        print(
            f"      Triangle: Lo={tri['ratio']:.2f}  "
            f"IC={tri['rank']:.3f}  Omega={tri['shape']:.2f}"
        )
        if r["failures"]:
            for f in r["failures"]:
                label = "SKIP" if verdict == "SKIP" else "FAIL"
                print(f"      {label}: {f}")
        for w in r.get("warnings", []):
            print(f"      WARN: {w}")

    n_pass = sum(1 for r in results.values() if r["verdict"] == "PASS")
    n_fail = sum(1 for r in results.values() if r["verdict"] == "FAIL")
    n_skip = sum(1 for r in results.values() if r["verdict"] == "SKIP")
    n_total = len(results)
    print(f"\n  {'=' * 66}")
    skip_note = f"  ({n_skip} skipped — run 'abel-edge run' first)" if n_skip else ""
    print(f"  {n_pass}/{n_total - n_skip} strategies pass Abel Proof validation{skip_note}")
    print("=" * 70)

    # ── Next steps (the product loop) ────────────────────────────────
    if n_fail > 0:
        print()
        print("  Next steps:")
        print("    Fix failures  → abel-edge validate --verbose")
        print("    Failure guide → abel_edge/validation/AGENTS.md")
        print("    Try your own  → docs/add-strategy.md")
        print("    Quick import  → abel-edge validate --csv your_backtest.csv")
    elif n_pass > 0 and n_fail == 0:
        print()
        print("  All strategies pass. Share your report card.")
        print("    Export → abel-edge validate --export report.txt")
