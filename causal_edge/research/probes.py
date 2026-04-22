"""Lightweight graph-node probes for agent-first exploration."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from causal_edge.graph_nodes import (
    GraphNodeRef,
    coerce_graph_node_refs,
    graph_node_assets,
    graph_node_runtime_field,
)
from causal_edge.plugins.abel.prices import fetch_bars

DEFAULT_PROBE_LIMIT = 500


def probe_graph_inputs(
    *,
    node_ids: list[str],
    target_node: str,
    start: str | None = None,
    end: str | None = None,
    limit: int = DEFAULT_PROBE_LIMIT,
    env_path: str = ".env",
) -> dict[str, Any]:
    target_refs = coerce_graph_node_refs([target_node])
    if not target_refs:
        raise ValueError("A valid --target-node is required.")
    target_ref = target_refs[0]
    requested_refs = coerce_graph_node_refs(node_ids)
    if not requested_refs:
        raise ValueError("At least one --node-id is required.")

    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for asset in graph_node_assets([target_ref, *requested_refs]):
        try:
            frame = fetch_bars(
                symbols=[asset],
                start=start,
                end=end,
                timeframe="1d",
                limit=limit,
                fields=["close", "volume"],
                config={"env_path": env_path},
            )
        except Exception as exc:  # pragma: no cover - exercised through rendered result shape
            errors[asset] = str(exc)
            frame = pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])
        if not frame.empty:
            frame = frame.copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            frame = frame.sort_values("timestamp").reset_index(drop=True)
        frames[asset] = frame

    target_series = _series_for_ref(frames.get(target_ref.asset), target_ref)
    target_index = pd.DatetimeIndex(target_series.index)
    sample_dates = _sample_dates(target_index)
    results = [
        _probe_result_for_ref(
            ref,
            frame=frames.get(ref.asset),
            target_index=target_index,
            sample_dates=sample_dates,
            requested_start=start,
            error=errors.get(ref.asset),
        )
        for ref in requested_refs
    ]
    return {
        "target": {
            "node_id": target_ref.node_id,
            "asset": target_ref.asset,
            "field": target_ref.field,
            "runtime_field": graph_node_runtime_field(target_ref),
            "row_count": int(len(target_series)),
            "decision_window": _window_from_index(target_index),
            "sample_decision_dates": [item.isoformat() for item in sample_dates],
        },
        "requested_window": {"start": start, "end": end},
        "probe": {"limit": int(limit), "timeframe": "1d"},
        "results": results,
        "basket": _basket_summary(results, target_index=target_index),
    }


def render_probe_report(report: dict[str, Any]) -> str:
    target = report.get("target") or {}
    basket = report.get("basket") or {}
    requested = report.get("requested_window") or {}
    lines = [
        "Graph Input Probe",
        f"Target node: {target.get('node_id', 'unknown')}",
        (
            "Requested window: "
            f"{requested.get('start') or 'latest'} -> {requested.get('end') or 'latest'}"
        ),
        (
            "Target decision window: "
            f"{((target.get('decision_window') or {}).get('start') or 'n/a')} -> "
            f"{((target.get('decision_window') or {}).get('end') or 'n/a')}"
        ),
        f"Sample decision dates: {', '.join(target.get('sample_decision_dates') or []) or 'none'}",
        (
            "Basket summary: "
            f"dense_overlap_start={basket.get('dense_overlap_start') or 'n/a'}, "
            f"fully_overlapping={basket.get('fully_overlapping_count', 0)}/{basket.get('probe_count', 0)}, "
            f"limiting_inputs={', '.join(basket.get('limiting_inputs') or []) or 'none'}"
        ),
        "",
    ]
    for item in report.get("results") or []:
        native = item.get("native_window") or {}
        lines.append(
            f"- {item.get('node_id')}: {item.get('status')} "
            f"(rows={item.get('row_count', 0)}, "
            f"native={native.get('start') or 'n/a'} -> {native.get('end') or 'n/a'}, "
            f"target_overlap={item.get('target_overlap_days', 0)}/{item.get('target_decision_days', 0)}, "
            f"field={item.get('field', 'unknown')})"
        )
    return "\n".join(lines)


def _probe_result_for_ref(
    ref: GraphNodeRef,
    *,
    frame: pd.DataFrame | None,
    target_index: pd.DatetimeIndex,
    sample_dates: list[pd.Timestamp],
    requested_start: str | None,
    error: str | None,
) -> dict[str, Any]:
    if error:
        return {
            "node_id": ref.node_id,
            "asset": ref.asset,
            "field": ref.field,
            "runtime_field": graph_node_runtime_field(ref),
            "status": "error",
            "row_count": 0,
            "native_window": {"start": None, "end": None},
            "target_overlap_days": 0,
            "target_decision_days": int(len(target_index)),
            "target_coverage_ratio": 0.0,
            "first_usable_target_time": None,
            "covers_requested_start": False,
            "native_sample": [],
            "asof_preview": [],
            "note": error,
        }

    series = _series_for_ref(frame, ref)
    native_index = pd.DatetimeIndex(series.index)
    aligned = _align_asof_to_target(series, target_index)
    overlap_mask = aligned.notna() if not aligned.empty else pd.Series(dtype=bool)
    overlap_days = int(overlap_mask.sum()) if not overlap_mask.empty else 0
    total_days = int(len(target_index))
    first_usable = None
    if not overlap_mask.empty and overlap_mask.any():
        first_usable = overlap_mask[overlap_mask].index[0]
    coverage_ratio = float(overlap_days / total_days) if total_days else 0.0
    status = _probe_status(series, overlap_days=overlap_days, total_days=total_days)
    native_window = _window_from_index(native_index)
    covers_requested_start = False
    if requested_start and native_window.get("start"):
        covers_requested_start = pd.Timestamp(native_window["start"]) <= pd.Timestamp(
            requested_start,
            tz="UTC",
        )
    if requested_start is None:
        covers_requested_start = bool(len(series) > 0)
    return {
        "node_id": ref.node_id,
        "asset": ref.asset,
        "field": ref.field,
        "runtime_field": graph_node_runtime_field(ref),
        "status": status,
        "row_count": int(len(series)),
        "native_window": native_window,
        "target_overlap_days": overlap_days,
        "target_decision_days": total_days,
        "target_coverage_ratio": coverage_ratio,
        "first_usable_target_time": first_usable.isoformat() if first_usable is not None else None,
        "covers_requested_start": covers_requested_start,
        "native_sample": _sample_series(series),
        "asof_preview": _sample_aligned(aligned, sample_dates),
        "note": _probe_note(status),
    }


def _series_for_ref(frame: pd.DataFrame | None, ref: GraphNodeRef) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float, name=ref.node_id)
    field = graph_node_runtime_field(ref)
    if field not in frame.columns:
        return pd.Series(dtype=float, name=ref.node_id)
    series = pd.Series(
        frame[field].astype(float).to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True)),
        name=ref.node_id,
    )
    return series.sort_index()


def _sample_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    if len(index) == 0:
        return []
    if len(index) <= 3:
        return list(index)
    anchors = sorted({0, len(index) // 2, len(index) - 1})
    return [pd.Timestamp(index[idx]) for idx in anchors]


def _sample_series(series: pd.Series, *, limit: int = 3) -> list[dict[str, Any]]:
    if series.empty:
        return []
    sampled = series.iloc[:limit]
    return [
        {"timestamp": pd.Timestamp(ts).isoformat(), "value": float(value)}
        for ts, value in sampled.items()
    ]


def _sample_aligned(series: pd.Series, sample_dates: list[pd.Timestamp]) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for ts in sample_dates:
        value = None
        if ts in series.index and pd.notna(series.loc[ts]):
            value = float(series.loc[ts])
        previews.append({"decision_time": pd.Timestamp(ts).isoformat(), "value": value})
    return previews


def _align_asof_to_target(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    if series.empty or len(target_index) == 0:
        return pd.Series(dtype=float, name=series.name)
    expanded = series.reindex(series.index.union(target_index)).sort_index().ffill()
    return expanded.reindex(target_index)


def _window_from_index(index: pd.DatetimeIndex) -> dict[str, Any]:
    if len(index) == 0:
        return {"start": None, "end": None}
    return {
        "start": pd.Timestamp(index.min()).isoformat(),
        "end": pd.Timestamp(index.max()).isoformat(),
    }


def _probe_status(series: pd.Series, *, overlap_days: int, total_days: int) -> str:
    if series.empty:
        return "no_data"
    if total_days == 0:
        return "target_unavailable"
    if overlap_days == total_days:
        return "full_target_overlap"
    if overlap_days > 0:
        return "partial_target_overlap"
    return "no_target_overlap"


def _probe_note(status: str) -> str:
    notes = {
        "no_data": "No native rows returned for this node in the requested window.",
        "target_unavailable": "The target produced no decision points in the requested window.",
        "full_target_overlap": "This node can be projected onto every target decision point.",
        "partial_target_overlap": "This node is usable on part of the target decision calendar only.",
        "no_target_overlap": "This node has native data but does not overlap the target decision calendar.",
    }
    return notes.get(status, status)


def _basket_summary(results: list[dict[str, Any]], *, target_index: pd.DatetimeIndex) -> dict[str, Any]:
    first_usable_times = [
        item.get("first_usable_target_time")
        for item in results
        if item.get("first_usable_target_time")
    ]
    latest_first = max(first_usable_times) if first_usable_times else None
    limiting = [
        item.get("node_id")
        for item in results
        if item.get("first_usable_target_time") == latest_first or not item.get("first_usable_target_time")
    ]
    return {
        "probe_count": len(results),
        "target_decision_days": int(len(target_index)),
        "fully_overlapping_count": sum(
            1 for item in results if item.get("status") == "full_target_overlap"
        ),
        "dense_overlap_start": latest_first,
        "limiting_inputs": [item for item in limiting if item],
    }


def availability_summary_from_probe_result(item: dict[str, Any]) -> dict[str, Any]:
    native_window = item.get("native_window") or {}
    return {
        "status": item.get("status"),
        "rows": int(item.get("row_count", 0) or 0),
        "start": native_window.get("start"),
        "end": native_window.get("end"),
        "target_overlap_days": int(item.get("target_overlap_days", 0) or 0),
        "target_decision_days": int(item.get("target_decision_days", 0) or 0),
        "first_usable_target_time": item.get("first_usable_target_time"),
    }


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)
