"""Agent-readable vocabulary for validation gate dimensions.

This module describes gate dimensions Abel-edge owns or accepts as validation
contract inputs. It does not choose a gate for a session and it does not inspect
strategy results.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from abel_edge import __version__

SCHEMA = "abel-edge.gate-vocabulary/v1"


def list_gate_vocabulary() -> dict[str, Any]:
    """Return deterministic, JSON-serializable gate vocabulary metadata."""

    dimensions = [_with_fingerprint(item) for item in _DIMENSIONS]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "edge_version": __version__,
        "dimensions": dimensions,
    }
    payload["vocabulary_hash"] = _stable_hash(payload)
    return payload


def gate_dimension_map() -> dict[str, dict[str, Any]]:
    """Return vocabulary entries keyed by dimension id."""

    return {str(item["id"]): item for item in list_gate_vocabulary()["dimensions"]}


def _with_fingerprint(item: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(item)
    copied["fingerprint"] = _stable_hash(copied)
    return copied


def _stable_hash(payload: dict[str, Any]) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


_DIMENSIONS: list[dict[str, Any]] = [
    {
        "id": "total_return",
        "display_name": "Total return",
        "meaning": "Cumulative compounded return over the evaluated window.",
        "unit": "fraction",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">", "value_type": "number"}],
        "numeric_meaning": (
            "Example: total_return > 0 requires positive cumulative return over the "
            "evaluated window."
        ),
        "deterministic_check_owner": "abel_edge.validation.metrics.compute_all_metrics",
        "source_metric": "total_return",
    },
    {
        "id": "annual_return",
        "display_name": "Annual return",
        "meaning": "Simple return annualized over the elapsed evaluation period.",
        "unit": "fraction_per_year",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">=", "value_type": "number"}],
        "numeric_meaning": (
            "Example: annual_return >= 0.05 requires at least about 5% simple "
            "annualized return."
        ),
        "deterministic_check_owner": "abel_edge.validation.metrics.compute_all_metrics",
        "source_metric": "annual_return",
    },
    {
        "id": "sharpe",
        "display_name": "Sharpe",
        "meaning": "Annualized mean return divided by return volatility.",
        "unit": "ratio",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">", "value_type": "number"}],
        "numeric_meaning": "Example: sharpe > 1.0 requires return above one unit of annualized volatility.",
        "deterministic_check_owner": "abel_edge.validation.metrics.compute_all_metrics",
        "source_metric": "sharpe",
    },
    {
        "id": "lo_adjusted",
        "display_name": "Lo-adjusted Sharpe",
        "meaning": "Sharpe ratio adjusted for first-order serial correlation.",
        "unit": "ratio",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">=", "value_type": "number"}],
        "numeric_meaning": (
            "Example: lo_adjusted >= 1.0 requires risk-adjusted return after "
            "autocorrelation adjustment."
        ),
        "deterministic_check_owner": "abel_edge.validation.gate_explain._lo_adjusted_check",
        "source_metric": "lo_adjusted",
    },
    {
        "id": "max_dd",
        "display_name": "Max drawdown",
        "meaning": "Signed peak-to-trough loss over the evaluated equity curve.",
        "unit": "fraction",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">=", "value_type": "number"}],
        "numeric_meaning": "Example: max_dd >= -0.15 rejects paths worse than roughly 15% drawdown.",
        "deterministic_check_owner": "abel_edge.validation.gate_explain._max_dd_check",
        "source_metric": "max_dd",
    },
    {
        "id": "dsr",
        "display_name": "Deflated Sharpe ratio",
        "meaning": "Probability-adjusted Sharpe evidence after accounting for multiple trials.",
        "unit": "probability",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">=", "value_type": "number"}],
        "numeric_meaning": (
            "Example: dsr >= 0.90 requires at least 90% deflated Sharpe evidence "
            "under the configured trial count."
        ),
        "deterministic_check_owner": "abel_edge.validation.gate_explain._dsr_check",
        "source_metric": "dsr",
    },
    {
        "id": "loss_years",
        "display_name": "Loss years",
        "meaning": "Count of full calendar years with negative total strategy return.",
        "unit": "count",
        "direction": "lower_is_better",
        "threshold_forms": [{"operator": "<=", "value_type": "integer"}],
        "numeric_meaning": "Example: loss_years <= 2 permits no more than two negative full calendar years.",
        "deterministic_check_owner": "abel_edge.validation.gate_explain._loss_years_checks",
        "source_metric": "loss_years",
    },
    {
        "id": "omega",
        "display_name": "Omega",
        "meaning": "Ratio of positive return mass to negative return mass for active returns.",
        "unit": "ratio",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">=", "value_type": "number"}],
        "numeric_meaning": "Example: omega >= 1.0 requires gains to at least match loss mass.",
        "deterministic_check_owner": "abel_edge.validation.gate_explain._omega_checks",
        "source_metric": "omega",
    },
    {
        "id": "return_floor",
        "display_name": "Return floor",
        "meaning": "Configured minimum total or annualized return required by the anti-gaming gate.",
        "unit": "fraction",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">=", "value_type": "number"}],
        "numeric_meaning": "Example: return_floor >= 0.01 requires return to clear a 1% configured floor.",
        "deterministic_check_owner": "abel_edge.validation.gate_explain._return_floor_check",
        "source_metric": "total_return_or_annual_return",
    },
    {
        "id": "sharpe_lo_ratio",
        "display_name": "Sharpe to Lo-adjusted ratio",
        "meaning": "Ratio used to detect Sharpe inflation relative to Lo-adjusted Sharpe.",
        "unit": "ratio",
        "direction": "lower_is_better",
        "threshold_forms": [{"operator": "<=", "value_type": "number"}],
        "numeric_meaning": (
            "Example: sharpe_lo_ratio <= 2.5 rejects cases where raw Sharpe is more "
            "than 2.5 times Lo-adjusted Sharpe."
        ),
        "deterministic_check_owner": "abel_edge.validation.gate_explain._sharpe_lo_ratio_check",
        "source_metric": "sharpe_lo_ratio",
    },
    {
        "id": "position_ic",
        "display_name": "Position IC",
        "meaning": "Spearman correlation between strategy position and next-period asset return.",
        "unit": "correlation",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">=", "value_type": "number"}],
        "numeric_meaning": (
            "Example: position_ic >= 0.02 requires positive rank correlation between "
            "position and subsequent return."
        ),
        "deterministic_check_owner": "abel_edge.validation.gate_explain._position_ic_checks",
        "source_metric": "position_ic",
    },
    {
        "id": "position_ic_stability",
        "display_name": "Position IC stability",
        "meaning": "Fractional stability measure for monthly position IC evidence.",
        "unit": "fraction",
        "direction": "higher_is_better",
        "threshold_forms": [{"operator": ">=", "value_type": "number"}],
        "numeric_meaning": (
            "Example: position_ic_stability >= 0.55 requires at least 55% stability "
            "under the configured monthly IC test."
        ),
        "deterministic_check_owner": "abel_edge.validation.gate_explain._position_ic_stability_checks",
        "source_metric": "position_ic_stability",
    },
    {
        "id": "edge_required_gates",
        "display_name": "Required Edge gates",
        "meaning": (
            "Meta-dimension requiring all applicable deterministic Abel-edge validation "
            "gates to pass."
        ),
        "unit": "boolean",
        "direction": "pass_required",
        "threshold_forms": [{"operator": "pass_all", "value_type": "boolean"}],
        "numeric_meaning": (
            "Example: pass_all true means every applicable deterministic Abel-edge "
            "validation check must pass."
        ),
        "deterministic_check_owner": "abel_edge.validation.gate_logic.validate",
        "source_metric": "validation_explanation.passed",
    },
    {
        "id": "position_bounds",
        "display_name": "Position bounds",
        "meaning": "Allowed lower and upper bounds for generated strategy positions.",
        "unit": "fractional_position",
        "direction": "within_bounds",
        "threshold_forms": [{"operator": "within", "value_type": "number_pair"}],
        "numeric_meaning": "Example: position_bounds within [0.0, 1.0] means long-only unlevered positions.",
        "deterministic_check_owner": "abel-edge runtime input contract",
        "source_metric": "strategy_position",
    },
    {
        "id": "search_width",
        "display_name": "Search width",
        "meaning": (
            "Audit dimension requiring the effective number of tried strategy variants "
            "to be recorded."
        ),
        "unit": "audit_count",
        "direction": "record_required",
        "threshold_forms": [{"operator": "recorded", "value_type": "boolean"}],
        "numeric_meaning": (
            "Example: recorded true means the effective number of tried variants is "
            "visible for research audit."
        ),
        "deterministic_check_owner": "abel-edge research audit contract",
        "source_metric": "selection_trials_or_search_width",
    },
]


__all__ = ["SCHEMA", "gate_dimension_map", "list_gate_vocabulary"]
