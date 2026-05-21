"""Validation gate decisions derived from computed metrics."""

from __future__ import annotations

from abel_edge.validation.gate_explain import explain_metric_gates as _explain_metric_gates


def validate(metrics: dict, profile: dict) -> tuple[bool, list[str]]:
    """Run validation gate. Returns (passed, list_of_failures)."""
    explanation = _explain_metric_gates(metrics, profile)
    return bool(explanation["passed"]), list(explanation["failures"])


def decide_keep_discard(current: dict, baseline: dict, profile: dict) -> str:
    """Metric triangle KEEP/DISCARD decision."""
    mt = profile.get("metric_triangle", {})
    v = profile.get("validation", {})

    opt_key = {
        "lo_adjusted_sharpe": "lo_adjusted",
        "sharpe": "sharpe",
        "total_return": "total_return",
    }.get(mt.get("optimize", "lo_adjusted_sharpe"), "lo_adjusted")
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
