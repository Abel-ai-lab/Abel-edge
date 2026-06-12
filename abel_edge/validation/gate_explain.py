"""Structured validation gate facts derived from computed metrics."""

from __future__ import annotations

from abel_edge.validation.gate_explain_common import check, explanation, finite_number


def explain_metric_gates(metrics: dict, profile: dict) -> dict:
    """Explain applicable metric validation gates as structured checks.

    The returned facts are derived from the same policy used by validate().
    This helper does not recompute metrics or inspect trade logs.
    """
    return _explain_standard_metric_gates(metrics, profile)


def _explain_standard_metric_gates(metrics: dict, profile: dict) -> dict:
    validation_cfg = profile.get("validation", {})
    anti_gaming = profile.get("anti_gaming", {})
    checks = [
        _dsr_check(metrics, validation_cfg),
        *_loss_years_checks(metrics, validation_cfg),
        _lo_adjusted_check(metrics, validation_cfg),
        *_omega_checks(metrics, validation_cfg),
        _max_dd_check(metrics, validation_cfg),
        _return_floor_check(metrics, anti_gaming),
        _sharpe_lo_ratio_check(metrics, anti_gaming),
        *_position_ic_checks(metrics, anti_gaming),
        *_position_ic_stability_checks(metrics, anti_gaming),
    ]
    return explanation(profile=profile, contract="standard", checks=checks)


def _dsr_check(metrics: dict, validation_cfg: dict) -> dict:
    dsr = metrics.get("dsr")
    threshold = float(validation_cfg.get("dsr_min", 0.90))
    if not finite_number(dsr):
        return check(
            check_id="dsr",
            metric="dsr",
            observed=None,
            threshold=threshold,
            comparison=">=",
            passed=False,
            message="T6 DSR invalid",
            invalid=True,
        )
    passed = float(dsr) >= threshold
    return check(
        check_id="dsr",
        metric="dsr",
        observed=float(dsr),
        threshold=threshold,
        comparison=">=",
        passed=passed,
        message=(
            f"T6 DSR {dsr:.1%} >= {threshold:.0%}"
            if passed
            else f"T6 DSR {dsr:.1%} < {threshold:.0%}"
        ),
    )


def _loss_years_checks(metrics: dict, validation_cfg: dict) -> list[dict]:
    if not metrics.get("loss_years_applicable", False):
        return []
    observed = int(metrics["loss_years"])
    threshold = int(validation_cfg.get("max_loss_years", 2))
    passed = observed <= threshold
    return [
        check(
            check_id="loss_years",
            metric="loss_years",
            observed=observed,
            threshold=threshold,
            comparison="<=",
            passed=passed,
            message=(
                f"T14 LossYrs {observed} <= {threshold}"
                if passed
                else f"T14 LossYrs {observed} > {threshold}"
            ),
        )
    ]


def _lo_adjusted_check(metrics: dict, validation_cfg: dict) -> dict:
    observed = float(metrics["lo_adjusted"])
    threshold = float(validation_cfg.get("lo_adjusted_min", 1.0))
    passed = observed >= threshold
    return check(
        check_id="lo_adjusted",
        metric="lo_adjusted",
        observed=observed,
        threshold=threshold,
        comparison=">=",
        passed=passed,
        message=(
            f"T15 Lo {observed:.2f} >= {threshold}"
            if passed
            else f"T15 Lo {observed:.2f} < {threshold}"
        ),
    )


def _omega_checks(metrics: dict, validation_cfg: dict) -> list[dict]:
    if not metrics.get("omega_applicable", False):
        return []
    observed = float(metrics["omega"])
    threshold = float(validation_cfg.get("omega_min", 1.0))
    passed = observed + 1e-12 >= threshold
    return [
        check(
            check_id="omega",
            metric="omega",
            observed=observed,
            threshold=threshold,
            comparison=">=",
            passed=passed,
            message=(
                f"T15 Omega {observed:.2f} >= {threshold}"
                if passed
                else f"T15 Omega {observed:.2f} < {threshold}"
            ),
        )
    ]


def _max_dd_check(metrics: dict, validation_cfg: dict) -> dict:
    observed = float(metrics["max_dd"])
    threshold = float(validation_cfg.get("max_dd", -0.20))
    passed = observed >= threshold
    return check(
        check_id="max_dd",
        metric="max_dd",
        observed=observed,
        threshold=threshold,
        comparison=">=",
        passed=passed,
        message=(
            f"T15 MaxDD {abs(observed) * 100:.1f}% <= {abs(threshold) * 100:.0f}%"
            if passed
            else f"T15 MaxDD {abs(observed) * 100:.1f}% > {abs(threshold) * 100:.0f}%"
        ),
    )


def _return_floor_check(metrics: dict, anti_gaming: dict) -> dict:
    threshold = float(anti_gaming.get("return_floor", 1.0))
    annualized = bool(anti_gaming.get("return_floor_annualized", False))
    if annualized:
        fallback = "annual_return" not in metrics
        observed_metric = "total_return" if fallback else "annual_return"
        observed = float(metrics.get("annual_return", metrics.get("total_return", 0.0)))
        metric = "annual_return"
        label = "Annualized return floor"
    else:
        fallback = False
        observed_metric = "total_return"
        observed = float(metrics["total_return"])
        metric = "total_return"
        label = "Return floor"
    passed = observed >= threshold
    return check(
        check_id="return_floor",
        metric=metric,
        observed=observed,
        threshold=threshold,
        comparison=">=",
        passed=passed,
        message=(
            f"{label} {observed * 100:+.2f}% >= +{threshold * 100:.2f}%"
            if passed
            else f"{label} {observed * 100:+.2f}% < +{threshold * 100:.2f}%"
        ),
        observed_metric=observed_metric,
        compatibility_fallback="annual_return_missing_total_return" if fallback else None,
    )


def _sharpe_lo_ratio_check(metrics: dict, anti_gaming: dict) -> dict:
    observed = float(metrics["sharpe_lo_ratio"])
    threshold = float(anti_gaming.get("sharpe_lo_ratio_max", 2.5))
    passed = not (
        metrics["sharpe"] > 0
        and metrics["lo_adjusted"] > 0
        and metrics["sharpe_lo_ratio"] > threshold
    )
    return check(
        check_id="sharpe_lo_ratio",
        metric="sharpe_lo_ratio",
        observed=observed,
        threshold=threshold,
        comparison="<=",
        passed=passed,
        message=(
            f"Sharpe/Lo {observed:.1f} <= {threshold}"
            if passed
            else f"Sharpe/Lo {observed:.1f} > {threshold}"
        ),
    )


def _position_ic_checks(metrics: dict, anti_gaming: dict) -> list[dict]:
    if not metrics.get("position_ic_applicable", False):
        return []
    observed = float(metrics["position_ic"])
    threshold = float(anti_gaming.get("position_ic_min", 0.02))
    passed = observed >= threshold
    return [
        check(
            check_id="position_ic",
            metric="position_ic",
            observed=observed,
            threshold=threshold,
            comparison=">=",
            passed=passed,
            message=(
                f"PositionIC {observed:.3f} >= {threshold}"
                if passed
                else f"PositionIC {observed:.3f} < {threshold}"
            ),
        )
    ]


def _position_ic_stability_checks(metrics: dict, anti_gaming: dict) -> list[dict]:
    if not metrics.get("position_ic_stability_applicable", False):
        return []
    observed = float(metrics["position_ic_stability"])
    threshold = float(anti_gaming.get("position_ic_stability_min", 0.55))
    passed = observed >= threshold
    return [
        check(
            check_id="position_ic_stability",
            metric="position_ic_stability",
            observed=observed,
            threshold=threshold,
            comparison=">=",
            passed=passed,
            message=(
                f"PositionIC stab {observed:.0%} >= {threshold:.0%}"
                if passed
                else f"PositionIC stab {observed:.0%} < {threshold:.0%}"
            ),
        )
    ]
