"""Validation gate decisions derived from computed metrics."""

from __future__ import annotations


def validate(metrics: dict, profile: dict) -> tuple[bool, list[str]]:
    """Run validation gate. Returns (passed, list_of_failures)."""
    v = profile.get("validation", {})
    ag = profile.get("anti_gaming", {})
    failures = []

    if metrics["dsr"] < v.get("dsr_min", 0.90):
        failures.append(f"T6 DSR {metrics['dsr']:.1%} < {v['dsr_min']:.0%}")
    if metrics["drawdown_time_frac"] > v.get("drawdown_time_frac_max", 0.35):
        failures.append(
            f"T13 DrawdownTime {metrics['drawdown_time_frac']:.0%} > {v['drawdown_time_frac_max']:.0%}"
        )
    if metrics.get("loss_years_applicable", False) and metrics["loss_years"] > v.get(
        "max_loss_years", 2
    ):
        failures.append(f"T14 LossYrs {metrics['loss_years']} > {v['max_loss_years']}")
    if metrics["lo_adjusted"] < v.get("lo_adjusted_min", 1.0):
        failures.append(f"T15 Lo {metrics['lo_adjusted']:.2f} < {v['lo_adjusted_min']}")
    if metrics.get("omega_applicable", False) and metrics["omega"] + 1e-12 < v.get(
        "omega_min", 1.0
    ):
        failures.append(f"T15 Omega {metrics['omega']:.2f} < {v['omega_min']}")
    if metrics["max_dd"] < v.get("max_dd", -0.20):
        failures.append(
            f"T15 MaxDD {abs(metrics['max_dd']) * 100:.1f}% > {abs(v['max_dd']) * 100:.0f}%"
        )
    if metrics["total_return"] < ag.get("return_floor", 1.0):
        failures.append(
            f"Return floor {metrics['total_return'] * 100:+.1f}% < +{ag['return_floor'] * 100:.0f}%"
        )
    if (
        metrics["sharpe"] > 0
        and metrics["lo_adjusted"] > 0
        and metrics["sharpe_lo_ratio"] > ag.get("sharpe_lo_ratio_max", 2.5)
    ):
        failures.append(
            f"Sharpe/Lo {metrics['sharpe_lo_ratio']:.1f} > {ag['sharpe_lo_ratio_max']}"
        )
    if metrics.get("position_ic_applicable", False) and metrics["position_ic"] < ag.get(
        "position_ic_min", 0.02
    ):
        failures.append(f"PositionIC {metrics['position_ic']:.3f} < {ag['position_ic_min']}")
    if metrics.get("position_ic_stability_applicable", False) and metrics[
        "position_ic_stability"
    ] < ag.get("position_ic_stability_min", 0.55):
        failures.append(
            f"PositionIC stab {metrics['position_ic_stability']:.0%} < {ag['position_ic_stability_min']:.0%}"
        )
    return len(failures) == 0, failures


def decide_keep_discard(current: dict, baseline: dict, profile: dict) -> str:
    """Metric triangle KEEP/DISCARD decision."""
    mt = profile.get("metric_triangle", {})
    v = profile.get("validation", {})

    opt_key = {"lo_adjusted_sharpe": "lo_adjusted", "sharpe": "sharpe"}.get(
        mt.get("optimize", "lo_adjusted_sharpe"), "lo_adjusted"
    )
    if current.get(opt_key, 0) <= baseline.get(opt_key, 0):
        return "DISCARD"

    for guard in mt.get("guardrails", []):
        key = {
            "raw_sharpe": "sharpe",
            "ic": "position_ic",
            "position_ic": "position_ic",
            "omega": "omega",
            "total_pnl": "total_return",
            "total_return": "total_return",
        }.get(guard["metric"], guard["metric"])
        tol = guard.get("tolerance", 0)
        if key == "total_return" and baseline.get(key, 0) > 0:
            if current.get(key, 0) < baseline[key] * (1 - tol):
                return "DISCARD"
        elif current.get(key, 0) < baseline.get(key, 0) - tol:
            return "DISCARD"

    if current.get("max_dd", 0) < v.get("max_dd", -0.25):
        return "DISCARD"

    return "KEEP"
