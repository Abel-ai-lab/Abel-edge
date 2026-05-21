"""Structured grandma validation gate facts."""

from __future__ import annotations

from abel_edge.validation.gate_explain_common import check, explanation, finite_number


def explain_grandma_metric_gates(metrics: dict, profile: dict) -> dict:
    validation_cfg = profile.get("validation", {})
    checks = [
        _grandma_return_check(metrics, validation_cfg),
        _grandma_pnl_to_maxdd_check(metrics, validation_cfg),
        _grandma_leverage_check(metrics, validation_cfg),
    ]
    return explanation(profile=profile, contract="grandma", checks=checks)


def _grandma_return_check(metrics: dict, validation_cfg: dict) -> dict:
    observed = metrics.get("total_return")
    threshold = float(validation_cfg.get("total_return_min", 0.0))
    if not finite_number(observed):
        return check(
            check_id="grandma_return",
            metric="total_return",
            observed=None,
            threshold=threshold,
            comparison=">",
            passed=False,
            message="Grandma return invalid",
            invalid=True,
        )
    observed = float(observed)
    passed = observed > threshold
    return check(
        check_id="grandma_return",
        metric="total_return",
        observed=observed,
        threshold=threshold,
        comparison=">",
        passed=passed,
        message=(
            f"Grandma return {observed * 100:+.2f}% > +{threshold * 100:.2f}%"
            if passed
            else f"Grandma return {observed * 100:+.2f}% <= +{threshold * 100:.2f}%"
        ),
    )


def _grandma_pnl_to_maxdd_check(metrics: dict, validation_cfg: dict) -> dict:
    observed = metrics.get("pnl_to_maxdd")
    threshold = float(validation_cfg.get("pnl_to_maxdd_min", 1.5))
    if not finite_number(observed):
        return check(
            check_id="grandma_pnl_to_maxdd",
            metric="pnl_to_maxdd",
            observed=None,
            threshold=threshold,
            comparison=">=",
            passed=False,
            message="Grandma PnL/MaxDD invalid",
            invalid=True,
        )
    observed = float(observed)
    passed = observed + 1e-12 >= threshold
    return check(
        check_id="grandma_pnl_to_maxdd",
        metric="pnl_to_maxdd",
        observed=observed,
        threshold=threshold,
        comparison=">=",
        passed=passed,
        message=(
            f"Grandma PnL/MaxDD {observed:.2f} >= {threshold:.2f}"
            if passed
            else f"Grandma PnL/MaxDD {observed:.2f} < {threshold:.2f}"
        ),
    )


def _grandma_leverage_check(metrics: dict, validation_cfg: dict) -> dict:
    threshold = float(validation_cfg.get("max_abs_position_max", 1.0))
    if not metrics.get("position_exposure_applicable", False):
        return check(
            check_id="grandma_max_abs_position",
            metric="max_abs_position",
            observed=None,
            threshold=threshold,
            comparison="<=",
            passed=False,
            message="Grandma leverage evidence missing position column",
            invalid=True,
        )

    observed = metrics.get("max_abs_position")
    if not finite_number(observed):
        return check(
            check_id="grandma_max_abs_position",
            metric="max_abs_position",
            observed=None,
            threshold=threshold,
            comparison="<=",
            passed=False,
            message="Grandma leverage invalid",
            invalid=True,
        )

    observed = float(observed)
    passed = observed <= threshold + 1e-12
    return check(
        check_id="grandma_max_abs_position",
        metric="max_abs_position",
        observed=observed,
        threshold=threshold,
        comparison="<=",
        passed=passed,
        message=(
            f"Grandma leverage {observed:.2f} <= {threshold:.2f}"
            if passed
            else f"Grandma leverage {observed:.2f} > {threshold:.2f}"
        ),
    )
