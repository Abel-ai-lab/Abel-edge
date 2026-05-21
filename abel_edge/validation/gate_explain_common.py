"""Shared helpers for structured validation gate facts."""

from __future__ import annotations

import math
from numbers import Real


def explanation(*, profile: dict, contract: str, checks: list[dict]) -> dict:
    failed_checks = [check for check in checks if check.get("applicable") and not check["passed"]]
    total_count = sum(1 for check in checks if check.get("applicable"))
    passed_count = total_count - len(failed_checks)
    failures = [str(check["message"]) for check in failed_checks]
    return {
        "profile": profile.get("name", "unknown"),
        "contract": contract,
        "passed": not failed_checks,
        "score": f"{passed_count}/{total_count}",
        "passed_count": passed_count,
        "total_count": total_count,
        "failures": failures,
        "failed_checks": failed_checks,
        "checks": checks,
    }


def check(
    *,
    check_id: str,
    metric: str,
    observed,
    threshold,
    comparison: str,
    passed: bool,
    message: str,
    observed_metric: str | None = None,
    compatibility_fallback: str | None = None,
    invalid: bool = False,
) -> dict:
    payload = {
        "id": check_id,
        "metric": metric,
        "observed_metric": observed_metric or metric,
        "observed": observed,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(passed),
        "applicable": True,
        "message": message,
    }
    if compatibility_fallback is not None:
        payload["compatibility_fallback"] = compatibility_fallback
    if invalid:
        payload["invalid"] = True
    return payload


def finite_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))
