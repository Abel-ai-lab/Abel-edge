"""Strategy admission gate — validate before adding to production.

Usage:
    from causal_edge.validation import validate_strategy
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

from causal_edge.validation.metrics import (
    compute_all_metrics,
    detect_profile,
    load_profile,
    validate,
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

    # PnL consistency check: |pnl - position * asset_return| should be small
    warnings: list[str] = []
    if positions is not None and asset_returns is not None:
        residuals = np.abs(pnl - positions * asset_returns)
        p95 = float(np.percentile(residuals, 95))
        if p95 >= 0.01:
            warnings.append(
                f"PnL consistency: 95th percentile of |pnl - position * asset_return| = {p95:.4f} "
                f"(threshold 0.01). Check for fees, slippage, or data errors."
            )

    # Auto-detect or load profile
    if profile is None:
        profile_name = detect_profile(pnl, dates, asset_returns=asset_returns)
    else:
        profile_name = profile
    prof = load_profile(profile_name)

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

    # Run validation gate
    passed, failures = validate(metrics, prof)

    # Extract triangle
    mt = prof.get("metric_triangle", {})
    opt_key = {"lo_adjusted_sharpe": "lo_adjusted", "sharpe": "sharpe"}.get(
        mt.get("optimize", "lo_adjusted_sharpe"), "lo_adjusted"
    )
    triangle = {
        "ratio": metrics.get(opt_key, 0),
        "rank": metrics.get("position_ic", 0),
        "shape": metrics.get("omega", 0),
    }

    # Count applicable tests
    total_tests = _count_total(metrics, prof)

    return {
        "verdict": "PASS" if passed else "FAIL",
        "score": f"{total_tests - len(failures)}/{total_tests}",
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
    from causal_edge.config import load_config

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
    skip_note = f"  ({n_skip} skipped — run 'causal-edge run' first)" if n_skip else ""
    print(f"  {n_pass}/{n_total - n_skip} strategies pass Abel Proof validation{skip_note}")
    print("=" * 70)

    # ── Next steps (the product loop) ────────────────────────────────
    if n_fail > 0:
        print()
        print("  Next steps:")
        print("    Fix failures  → causal-edge validate --verbose")
        print("    Failure guide → causal_edge/validation/AGENTS.md")
        print("    Try your own  → docs/add-strategy.md")
        print("    Quick import  → causal-edge validate --csv your_backtest.csv")
    elif n_pass > 0 and n_fail == 0:
        print()
        print("  All strategies pass. Share your report card.")
        print("    Export → causal-edge validate --export report.txt")


def _count_total(metrics: dict, profile: dict) -> int:
    """Count total applicable validation checks."""
    # Drawdown-time metrics remain diagnostic-only and no longer add gate slots.
    count = 4  # DSR, Lo, MaxDD, PnL floor
    if metrics.get("loss_years_applicable", False):
        count += 1  # LossYrs
    if metrics.get("omega_applicable", False):
        count += 1  # Omega
    count += 1  # Sharpe/Lo ratio
    if metrics.get("position_ic_applicable", False):
        count += 1  # PositionIC min
    if metrics.get("position_ic_stability_applicable", False):
        count += 1  # PositionIC stability
    return count
