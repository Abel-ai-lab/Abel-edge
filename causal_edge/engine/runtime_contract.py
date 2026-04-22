"""Agent-first runtime contract helpers for strategy engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from causal_edge.engine.backtest import compile_effective_positions
from causal_edge.engine.feed_contract import FeedContractError, validate_datetime_index
from causal_edge.graph_nodes import normalize_graph_node_id


class DecisionContractError(FeedContractError):
    """Raised when a decision-contract strategy violates the runtime contract."""


@dataclass(frozen=True)
class RuntimeProfile:
    """System-owned description of the execution clock for a run."""

    profile: str
    target: str | None
    target_node: str | None = None
    decision_event: str = "bar_close"
    execution_delay_bars: int = 1
    return_basis: str = "close_to_close"


@dataclass(frozen=True)
class ExecutionConstraints:
    """System-owned execution envelope for a run."""

    position_bounds: tuple[float, float] = (-np.inf, np.inf)
    long_only: bool = False


@dataclass(frozen=True)
class DecisionDraft:
    """Strategy-owned next-position intent over a decision index."""

    decision_index: pd.DatetimeIndex
    next_position: np.ndarray


@dataclass(frozen=True)
class CompiledDecisionOutput:
    """Runtime-owned compiled strategy output.

    ``positions`` are the effective exposures seen by the backtest kernel.
    ``next_position`` keeps the branch-authored intent so downstream tooling can
    surface both meanings during the rollout.
    """

    decision_index: pd.DatetimeIndex
    close_prices: np.ndarray
    positions: np.ndarray
    next_position: np.ndarray
    runtime_profile: RuntimeProfile
    execution_constraints: ExecutionConstraints
    output_mode: str


def runtime_profile_from_context(context: dict | None) -> RuntimeProfile:
    payload = ((context or {}).get("_runtime_profile") or {}) if isinstance(context, dict) else {}
    data_contract = ((context or {}).get("_data_contract") or {}) if isinstance(context, dict) else {}
    profile = str(payload.get("profile") or data_contract.get("profile") or "daily").strip().lower()
    target_node = _normalize_target_node(
        payload.get("target_node")
        or ((context or {}).get("branch_spec") or {}).get("target_node")
    )
    target = _normalize_target(
        payload.get("target_asset")
        or payload.get("target")
        or (target_node.rpartition(".")[0] if target_node else None)
        or (context or {}).get("asset")
        or (context or {}).get("ticker")
        or ((context or {}).get("discovery") or {}).get("ticker")
        or ((context or {}).get("branch_spec") or {}).get("target_asset")
        or ((context or {}).get("branch_spec") or {}).get("target")
    )
    decision_event = str(payload.get("decision_event") or "bar_close").strip().lower() or "bar_close"
    return_basis = str(payload.get("return_basis") or "close_to_close").strip().lower()
    execution_delay_bars = int(payload.get("execution_delay_bars", 1) or 0)
    if execution_delay_bars < 0:
        raise DecisionContractError("execution_delay_bars must be >= 0.")
    return RuntimeProfile(
        profile=profile,
        target=target,
        target_node=target_node,
        decision_event=decision_event,
        execution_delay_bars=execution_delay_bars,
        return_basis=return_basis or "close_to_close",
    )


def execution_constraints_from_context(context: dict | None) -> ExecutionConstraints:
    payload = ((context or {}).get("_execution_constraints") or {}) if isinstance(context, dict) else {}
    bounds = payload.get("position_bounds")
    if bounds is None:
        max_abs = payload.get("max_abs_position")
        if max_abs is not None:
            max_value = float(max_abs)
            bounds = (-max_value, max_value)
        else:
            bounds = (-np.inf, np.inf)
    lower, upper = _normalize_position_bounds(bounds)
    long_only = bool(payload.get("long_only", False))
    if long_only and lower < 0:
        lower = 0.0
    return ExecutionConstraints(position_bounds=(lower, upper), long_only=long_only)


def build_decision_draft(
    decision_index,
    next_position,
    *,
    runtime_profile: RuntimeProfile,
) -> DecisionDraft:
    try:
        index = validate_datetime_index(
            decision_index,
            profile=runtime_profile.profile,
            name="decision_index",
        )
    except FeedContractError as exc:
        raise DecisionContractError(str(exc)) from exc
    next_pos = _as_numeric_vector(next_position, name="next_position")
    if len(index) != len(next_pos):
        raise DecisionContractError(
            "decision_index and next_position must have identical lengths."
        )
    return DecisionDraft(decision_index=index, next_position=next_pos)


def compile_decision_draft(
    draft: DecisionDraft,
    close_prices,
    *,
    runtime_profile: RuntimeProfile,
    execution_constraints: ExecutionConstraints,
    output_mode: str = "decision_context",
) -> CompiledDecisionOutput:
    price_arr = _as_numeric_vector(close_prices, name="close_prices")
    if len(price_arr) != len(draft.decision_index):
        raise DecisionContractError(
            "close_prices and decision_index must have identical lengths."
        )

    next_position = draft.next_position.copy()
    lower, upper = execution_constraints.position_bounds
    next_position = np.clip(next_position, lower, upper)
    if execution_constraints.long_only:
        next_position = np.maximum(next_position, 0.0)

    max_abs_position = None if np.isinf([abs(lower), abs(upper)]).all() else float(max(abs(lower), abs(upper)))
    positions, next_position = compile_effective_positions(
        next_position,
        execution_delay_bars=runtime_profile.execution_delay_bars,
        max_abs_position=max_abs_position,
    )

    return CompiledDecisionOutput(
        decision_index=draft.decision_index,
        close_prices=price_arr,
        positions=positions,
        next_position=next_position,
        runtime_profile=runtime_profile,
        execution_constraints=execution_constraints,
        output_mode=output_mode,
    )


def legacy_output_to_compiled(
    positions,
    dates,
    prices,
    *,
    runtime_profile: RuntimeProfile,
    execution_constraints: ExecutionConstraints,
) -> CompiledDecisionOutput:
    try:
        index = validate_datetime_index(
            dates,
            profile=runtime_profile.profile,
            name="dates",
        )
    except FeedContractError as exc:
        raise DecisionContractError(str(exc)) from exc
    position_arr = _as_numeric_vector(positions, name="positions")
    price_arr = _as_numeric_vector(prices, name="prices")
    if len(index) != len(position_arr) or len(index) != len(price_arr):
        raise DecisionContractError("positions, dates, and prices must have identical lengths.")
    lower, upper = execution_constraints.position_bounds
    clipped = np.clip(position_arr, lower, upper)
    if execution_constraints.long_only:
        clipped = np.maximum(clipped, 0.0)
    return CompiledDecisionOutput(
        decision_index=index,
        close_prices=price_arr,
        positions=clipped,
        next_position=clipped.copy(),
        runtime_profile=runtime_profile,
        execution_constraints=execution_constraints,
        output_mode="legacy_signal_contract",
    )


def _as_numeric_vector(values, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise DecisionContractError(f"{name} must be a 1D numeric array.")
    if not np.isfinite(arr).all():
        raise DecisionContractError(f"{name} contains non-finite numeric values.")
    return arr


def _normalize_target(value: Any) -> str | None:
    target = str(value or "").strip().upper()
    return target or None


def _normalize_target_node(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return normalize_graph_node_id(raw)


def _normalize_position_bounds(bounds: Any) -> tuple[float, float]:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise DecisionContractError(
            "position_bounds must be a 2-item list/tuple like [-1.0, 1.0]."
        )
    lower = float(bounds[0])
    upper = float(bounds[1])
    if np.isnan(lower) or np.isnan(upper):
        raise DecisionContractError("position_bounds cannot contain NaN values.")
    if lower > upper:
        raise DecisionContractError("position_bounds lower bound cannot exceed upper bound.")
    return lower, upper
