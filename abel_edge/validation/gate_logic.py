"""Validation gate decisions derived from computed metrics."""

from __future__ import annotations

import math
from numbers import Real


def validate(metrics: dict, profile: dict) -> tuple[bool, list[str]]:
    """Run validation gate. Returns (passed, list_of_failures)."""
    v = profile.get("validation", {})
    if v.get("contract") == "grandma":
        return _validate_grandma(metrics, v)

    ag = profile.get("anti_gaming", {})
    failures = []

    dsr = metrics.get("dsr")
    if isinstance(dsr, bool) or not isinstance(dsr, Real) or not math.isfinite(float(dsr)):
        failures.append("T6 DSR invalid")
    elif dsr < v.get("dsr_min", 0.90):
        failures.append(f"T6 DSR {metrics['dsr']:.1%} < {v['dsr_min']:.0%}")
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
    return_floor = ag.get("return_floor", 1.0)
    if ag.get("return_floor_annualized", False):
        observed_return = metrics.get("annual_return", metrics.get("total_return", 0.0))
        failure_label = "Annualized return floor"
    else:
        observed_return = metrics["total_return"]
        failure_label = "Return floor"
    if observed_return < return_floor:
        failures.append(
            f"{failure_label} {observed_return * 100:+.2f}% < +{return_floor * 100:.2f}%"
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

    opt_key = {"lo_adjusted_sharpe": "lo_adjusted", "sharpe": "sharpe", "total_return": "total_return"}.get(
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


def _validate_grandma(metrics: dict, validation_cfg: dict) -> tuple[bool, list[str]]:
    failures = []
    total_return = metrics.get("total_return")
    min_return = float(validation_cfg.get("total_return_min", 0.0))
    if isinstance(total_return, bool) or not isinstance(total_return, Real) or not math.isfinite(float(total_return)):
        failures.append("Grandma return invalid")
    elif float(total_return) <= min_return:
        failures.append(f"Grandma return {float(total_return) * 100:+.2f}% <= +{min_return * 100:.2f}%")

    ratio = metrics.get("pnl_to_maxdd")
    ratio_min = float(validation_cfg.get("pnl_to_maxdd_min", 1.5))
    if isinstance(ratio, bool) or not isinstance(ratio, Real) or not math.isfinite(float(ratio)):
        failures.append("Grandma PnL/MaxDD invalid")
    elif float(ratio) + 1e-12 < ratio_min:
        failures.append(f"Grandma PnL/MaxDD {float(ratio):.2f} < {ratio_min:.2f}")

    max_position = float(validation_cfg.get("max_abs_position_max", 1.0))
    if not metrics.get("position_exposure_applicable", False):
        failures.append("Grandma leverage evidence missing position column")
    else:
        observed = metrics.get("max_abs_position")
        if isinstance(observed, bool) or not isinstance(observed, Real) or not math.isfinite(float(observed)):
            failures.append("Grandma leverage invalid")
        elif float(observed) > max_position + 1e-12:
            failures.append(f"Grandma leverage {float(observed):.2f} > {max_position:.2f}")

    return len(failures) == 0, failures
